import pandas as pd
import logging
import os
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.schema import Base, User
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf
from sklearn.preprocessing import OneHotEncoder
from rapidfuzz import fuzz
from joblib import Parallel, delayed,parallel_backend
import joblib
from multiprocessing import Manager
import json
from functools import lru_cache
import ast


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

log_file_path = os.path.join('logs', 'app.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path)
    ]
)

logger = logging.getLogger(__name__)

# SQLAlchemy engine and session setup
DATABASE_URL = "sqlite:///users.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

class UserAnalysis:
    def __init__(self, csv_file='data/random_users.csv', output_file='notebooks/user_analysis.ipynb'):
        self.csv_file = csv_file
        self.session = Session()
        self.output_file = output_file
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def create_table(self):
        """Create the database table using SQLAlchemy models."""
        try:
            User.__table__.drop(engine)  # Drop the existing table
            logger.info("Dropped existing table.")
            Base.metadata.create_all(engine)
            logger.info("Table created (if not already exists).")
        except Exception as e:
            logger.error(f"Error during table creation: {e}")

    def load_data(self):
        """Load data from the CSV file into the database."""
        try:
            logger.info(f"Loading data from {self.csv_file}")
            df = pd.read_csv(self.csv_file)
            logger.info(f"Data loaded from {self.csv_file} - {len(df)} records.")
            users = [
                User(
                    uid=row['uid'],
                    password=row['password'],
                    first_name=row['first_name'],
                    last_name=row['last_name'],
                    username=row['username'],
                    email=row['email'],
                    avatar=row['avatar'],
                    gender=row['gender'],
                    phone_number=row['phone_number'],
                    social_insurance_number=row['social_insurance_number'],
                    date_of_birth=row['date_of_birth'],
                    employment=row['employment'],
                    address=row['address'],
                    credit_card=row['credit_card'],
                    subscription=row['subscription']
                )
                for _, row in df.iterrows()
            ]
            self.session.bulk_save_objects(users)
            self.session.commit()
            logger.info("Data loaded into the database.")
        except Exception as e:
            logger.error(f"Error during data loading: {e}")


    def find_most_common_properties(self, threshold=3):
        """Find the most common properties across users."""
        try:
            users = self.session.query(User).all()

            # Prepare data for analysis (e.g., gender, employment, subscription)
            properties = {
                'gender': [user.gender for user in users],
                'employment': [self.simplify_employment(user.employment) for user in users],
                'subscription': [self.simplify_subscription(user.subscription) for user in users],
                'address': [self.simplify_address(user.address) for user in users],
                # 'phone_number': [user.phone_number for user in users],
                # 'social_insurance_number': [user.social_insurance_number for user in users],
                # 'date_of_birth': [user.date_of_birth for user in users],
            }

            # Analyze frequency of each property
            common_properties = {}
            for prop, values in properties.items():
                counter = Counter(values)
                # Get the most common items and filter out those with counts less than the threshold
                common_items = counter.most_common()
                filtered_common_items = [(value, count) for value, count in common_items if count >= threshold]

                common_properties[prop] = filtered_common_items

            logger.info(f"Most common properties")
            return common_properties
        except Exception as e:
            logger.error(f"Error during finding common properties: {e}")

    # Helper methods to simplify complex fields
    @lru_cache(maxsize=1000)
    def simplify_employment(self, employment_data):
        # Extract the key skill or title from employment (you can choose which value you want to keep)
        # employment_data is a string representing a dictionary
        if isinstance(employment_data, str):
            employment_data = ast.literal_eval(employment_data)
        return employment_data.get('title', '')

    @lru_cache(maxsize=1000)
    def simplify_subscription(self, subscription_data):
        # Extract the plan or the status from subscription
        if isinstance(subscription_data, str):
            subscription_data = ast.literal_eval(subscription_data)
        return subscription_data.get('plan', '')

    @lru_cache(maxsize=1000)
    def simplify_address(self, address_data):
        # Extract the city or street name from the address
        if isinstance(address_data, str):
            address_data = ast.literal_eval(address_data)
        return address_data.get('state', '')

    def find_similarities_between_users(self):
        """Find similarities between users based on different properties."""
        try:
            users = self.session.query(User).all()

            # Initialize OneHotEncoder
            encoder = OneHotEncoder(sparse=False)

            # Prepare lists to collect user properties
            gender_list = []
            employment_list = []
            subscription_list = []
            address_list = []

            # Collect user data
            for user in users:
                gender_list.append(str(user.gender) if user.gender else "Unknown")
                employment_list.append(self.simplify_employment(user.employment))
                subscription_list.append(self.simplify_subscription(user.subscription))
                address_list.append(self.simplify_address(user.address))

            # One-hot encode each list (each list represents one feature)
            gender_encoded = encoder.fit_transform([[gender] for gender in gender_list])
            employment_encoded = encoder.fit_transform([[employment] for employment in employment_list])
            subscription_encoded = encoder.fit_transform([[subscription] for subscription in subscription_list])
            address_encoded = encoder.fit_transform([[address] for address in address_list])

            # Concatenate all the encoded features into one matrix
            user_properties = np.hstack([gender_encoded, employment_encoded, subscription_encoded, address_encoded])

            # Scale the data
            scaler = StandardScaler()
            user_properties_scaled = scaler.fit_transform(user_properties)

            # Compute pairwise similarities between users using cosine similarity
            similarities = cosine_similarity(user_properties_scaled)

            # Log the similarity matrix
            logger.debug(f"Similarity Matrix: {similarities}")

            # Create a graph for different pairwise combinations
            G_gender_employment = self.create_graph_from_similarity(similarities, gender_list, employment_list)
            G_address_subscription = self.create_graph_from_similarity(similarities, address_list, subscription_list)
            G_gender_subscription = self.create_graph_from_similarity(similarities, gender_list, subscription_list)
            G_address_gender_employment = self.create_graph_from_similarity(similarities, address_list, gender_list, employment_list)

            return G_gender_employment, G_address_subscription, G_gender_subscription, G_address_gender_employment

        except Exception as e:
            logger.error(f"Error during finding similarities between users: {e}")
            return nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()  # Return empty graphs on error

    def create_graph_from_similarity(self, similarities, *properties):
        """Create a graph based on similarities and the chosen properties."""
        G = nx.Graph()
        logger.info(f"Creating graph from similarity matrix ")
        
        for i in range(len(properties[0])):  # Assuming all lists are of equal length
            for j in range(i + 1, len(properties[0])):
                similarity_score = similarities[i][j]
                if similarity_score > 0.5:  # Define a threshold for a strong connection
                    G.add_edge(i, j, weight=similarity_score, type='strong')
                elif similarity_score > 0.1:  # Define a threshold for a weak connection
                    G.add_edge(i, j, weight=similarity_score, type='weak')

        return G
    
    
    # Step 1: Optimized Similar Users Function
    def fetch_user_data(self, user_id):
        """Fetch relevant user data without passing the session explicitly."""
        
        # Create a new session for each request if needed
        session = Session()  # Start a new session for the current request.
        
        try:
            # Assuming you need the session to fetch user data, this fetches only relevant info
            user = session.query(User).filter(User.id == user_id).first()
            
            # Check if user was found
            if user:
                return {
                    "username": user.username,
                    "gender": user.gender,
                    "employment": user.employment,
                    "subscription": user.subscription,
                    "address": user.address,
                }
            else:
                return None  # Return None if user not found
        finally:
            session.close()  # Make sure to close the session after the operation
        
    def preprocess_user(self, user):
        """Preprocess user fields for faster similarity comparisons."""
        return {
            "username": user['username'],
            "gender": user['gender'],
            "employment": self.simplify_employment(user['employment']),
            "subscription": self.simplify_subscription(user['subscription']),
            "address": self.simplify_address(user['address']),
        }
        
    def calculate_fuzzy_similarity(self,value1, value2, thresholds=(90, 50)):
        """
        Calculate fuzzy similarity between two strings and return 'strong', 'weak', or 'none'.
        
        Parameters:
        - value1: First value (string).
        - value2: Second value (string).
        - thresholds: Tuple of thresholds for 'strong' and 'weak' similarities.
        
        Returns:
        - 'strong', 'weak', or 'none'.
        """
        similarity = fuzz.ratio(value1, value2)
        if similarity > thresholds[0]:
            return "strong"
        elif similarity > thresholds[1]:
            return "weak"
        return "none"

    def calculate_similarity(self, user1_data, user2_data):
        """Calculate similarity between two users."""
        
        similarities = {}
        
        try:
            # Preprocess user data
            user1 = self.preprocess_user(user1_data)
            user2 = self.preprocess_user(user2_data)
            
            # Gender similarity (can be direct comparison)
            similarities["gender"] = "strong" if user1["gender"] == user2["gender"] else "none"
            
            # Using the helper function for fuzzy similarities
            similarities["username"] = self.calculate_fuzzy_similarity(user1["username"], user2["username"])
            similarities["employment"] = self.calculate_fuzzy_similarity(user1["employment"], user2["employment"])
            similarities["subscription"] = self.calculate_fuzzy_similarity(user1["subscription"], user2["subscription"])
            similarities["address"] = self.calculate_fuzzy_similarity(user1["address"], user2["address"])

            # Check if any similarity is significant
            if any(value in ["strong", "weak"] for value in similarities.values()):
                return {
                    "user1": user1["username"],
                    "user2": user2["username"],
                    "similarities": similarities,
                }
            return None

        except Exception as e:
            logger.error(f"Error calculating similarity between {user1_data['username']} and {user2_data['username']}: {e}", exc_info=True)
            return None
    def find_similar_users_optimized(self, users):
        """Optimized function to find similar users without passing session directly to Parallel."""
        
        logger.info("Starting optimized similarity analysis.")
        
        try:
            # Step 1: Fetch and preprocess user data outside of the parallel execution
            logger.debug("Fetching user data for all users.")
            preprocessed_users = [self.fetch_user_data(user.id) for user in users]
            logger.info(f"Fetched and serialized user data for {type(preprocessed_users)} users.")
            

            
            # Step 2: Parallelize the pairwise comparisons using preprocessed data
            logger.debug("Starting pairwise similarity calculations.")
            
            # Use Manager to create a shared counter
            with Manager() as manager:
                progress = manager.Value('i', 0)  # Shared counter for progress

                # Function to update progress
                def update_progress(i, total):
                    progress.value += 1
                    # Log progress intermittently during processing
                    if progress.value % 100 == 0 or progress.value == total:
                        self.log_progress(progress.value, total)

                # Function to calculate similarity and update progress
                def log_progress_and_calculate_similarity(i, j):
                    pair = self.calculate_similarity(preprocessed_users[i], preprocessed_users[j])
                    # Update and log progress after every task
                    update_progress(i, len(preprocessed_users) * (len(preprocessed_users) - 1) // 2)
                    return pair

                # Parallelize the pairwise comparisons using preprocessed data
                similar_pairs = Parallel(n_jobs=-1, backend='threading')(
                    delayed(log_progress_and_calculate_similarity)(i, j)
                    for i in range(len(preprocessed_users))
                    for j in range(i + 1, len(preprocessed_users))
                )
                
                # batch_size = 1000
                # similar_pairs = []
                # for i in range(0, len(preprocessed_users), batch_size):
                #     batch = [
                #         delayed(log_progress_and_calculate_similarity)(i, j)
                #         for i in range(i, min(i + batch_size, len(preprocessed_users)))
                #         for j in range(i + 1, len(preprocessed_users))
                #     ]
                #     similar_pairs.extend(Parallel(n_jobs=-1, backend='threading')(batch))
                
                
                # Filter out None values
                similar_pairs = [pair for pair in similar_pairs if pair]
                logger.info(f"Found {len(similar_pairs)} similar user pairs.")
                return similar_pairs
            
        except Exception as e:
            logger.error(f"Error in optimized similarity analysis: {e}", exc_info=True)

        
            
    # def find_similar_users_optimized(self, users):
    #     """Optimized function to find similar users without passing session directly to Parallel."""
        
    #     logger.info("Starting optimized similarity analysis.")
        
    #     try:
    #         # Prepare user data in advance outside of the parallel execution
    #         logger.debug("Fetching user data for all users.")
    #         preprocessed_users = [self.fetch_user_data(user.id) for user in users]
    #         logger.info(f"Fetched user data for {len(preprocessed_users)} users.")
            
    #         # Parallelize the pairwise comparisons using preprocessed data
    #         logger.debug("Starting pairwise similarity calculations.")
            
    #         # Use Manager to create a shared counter
    #         with Manager() as manager:
    #             progress = manager.Value('i', 0)  # Shared counter for progress

    #             # Function to update progress
    #             def update_progress(i, total):
    #                 progress.value += 1
    #                 # Log progress intermittently during processing
    #                 if progress.value % 100 == 0 or progress.value == total:
    #                     self.log_progress(progress.value, total)

    #             # Function to calculate similarity and update progress
    #             def log_progress_and_calculate_similarity(i, j):
    #                 pair = self.calculate_similarity(preprocessed_users[i], preprocessed_users[j])
    #                 # Update and log progress after every task
    #                 update_progress(i, len(preprocessed_users) * (len(preprocessed_users) - 1) // 2)
    #                 return pair

    #             # Parallelize the pairwise comparisons using preprocessed data
    #             similar_pairs = Parallel(n_jobs=-1, backend='loky')(
    #                 delayed(log_progress_and_calculate_similarity)(i, j)
    #                 for i in range(len(preprocessed_users))
    #                 for j in range(i + 1, len(preprocessed_users))
    #             )
                
    #             # Filter out None values
    #             similar_pairs = [pair for pair in similar_pairs if pair]
    #             logger.info(f"Found {len(similar_pairs)} similar user pairs.")
    #             return similar_pairs
            
    #     except Exception as e:
    #         logger.error(f"Error in optimized similarity analysis: {e}", exc_info=True)

    def log_progress(self, completed, total):
        """Log the progress of the task."""
        logger.info(f"Progress: {completed}/{total} tasks completed.")
    def create_similarity_dataframe(self, similar_users):
        """Create a DataFrame from the similar user data."""
        data = [
            {
                "User 1": pair["user1"],
                "User 2": pair["user2"],
                "Username Similarity": pair["similarities"]["username"],
                "Gender Match": pair["similarities"]["gender"],
                "Employment Similarity": pair["similarities"]["employment"],
                "Subscription Similarity": pair["similarities"]["subscription"],
                "Address Similarity": pair["similarities"]["address"],
            }
            for pair in similar_users
        ]
        df = pd.DataFrame(data)
        
        # Add Relationship Strength column
        df["Relationship Strength"] = df.apply(
            lambda row: self.determine_relationship_strength({
                "username": row["Username Similarity"],
                "employment": row["Employment Similarity"],
                "subscription": row["Subscription Similarity"],
                "address": row["Address Similarity"],
                "gender": row["Gender Match"],
            }),
            axis=1,
        )
        return df
    
    def determine_relationship_strength(self, similarities):
        """Determine the relationship strength based on the similarity values."""
        if "strong" in similarities.values():
            return "strong"
        elif "weak" in similarities.values():
            return "weak"
        return "none"
    
    def add_visualization(self, nb, title, column, palette):
        """Helper function to add a visualization to the notebook."""
        nb.cells.append(nbf.v4.new_markdown_cell(f"### {title}"))
        nb.cells.append(
            nbf.v4.new_code_cell(
                f"""
                plt.figure(figsize=(8, 5))
                sns.countplot(data=similar_users_df, x="{column}", palette="{palette}")
                plt.title('{title}')
                plt.xlabel('{column}')
                plt.ylabel('Count')
                plt.show()
                """
            )
        )
    
    def generate_similarity_notebook(self, users, session, nb):
        """Generate a Jupyter notebook for user similarity analysis."""
        
        logger.info("Starting similarity analysis notebook generation.")
        
        try:
            # Find similar users
            logger.debug("Finding similar users using optimized method.")
            similar_users = self.find_similar_users_optimized(users)
            logger.info(f"Found {len(similar_users)} similar users.")

            # Create similarity DataFrame
            logger.debug("Creating similarity DataFrame.")
            similar_users_df = self.create_similarity_dataframe(similar_users)
            
            
            file_path = 'notebooks/similar_users_df.csv'
            file_location="similar_users_df.csv"

            
            # Log the process of saving the DataFrame
            logger.debug(f"Saving similarity DataFrame to {file_path}.")
            similar_users_df.to_csv(file_path, index=False)
            
            
            nb.cells.append(nbf.v4.new_code_cell(f"""
                # Load the similarity DataFrame
                similar_users_df = pd.read_csv('{file_location}')
                
                # Preview the first few rows of the DataFrame
                similar_users_df.head()
            """))

            logger.info("Similarity DataFrame created successfully.")

            # # Add DataFrame preview code
            # logger.debug("Adding DataFrame preview to notebook.")
            # nb.cells.append(nbf.v4.new_code_cell(f"""
            #     # Preview similar users DataFrame
            #     similar_users_df = {similar_users_df_repr}
            #     similar_users_df.head()
            # """))
            logger.info("DataFrame preview added to notebook.")

            # Add visualizations
            logger.debug("Adding visualizations to notebook.")
            visualizations = [
                ("Distribution of Relationship Strength", "Relationship Strength", "coolwarm"),
                ("Username Similarity Distribution", "Username Similarity", "Set2"),
                ("Gender Match Distribution", "Gender Match", "pastel"),
                ("Subscription Similarity Distribution", "Subscription Similarity", "muted"),
                ("Address Similarity Distribution", "Address Similarity", "cool"),
            ]
            
            for title, column, palette in visualizations:
                logger.debug(f"Adding visualization for {title} based on column {column}.")
                self.add_visualization(nb, title, column, palette)
            
            logger.info("Visualizations added successfully.")
        
        except Exception as e:
            logger.error(f"Error generating similarity analysis notebook: {e}", exc_info=True)

    def close_session(self):
        """Close the SQLAlchemy session."""
        self.session.close()
        logger.info("Database session closed.")
        
           
    
    # Function to generate notebook
    def create_notebook(self, common_properties, G_gender_employment, G_address_subscription, G_gender_subscription, G_address_gender_employment):
        logger.info("Starting notebook creation.")

        # Create a new notebook
        nb = nbf.v4.new_notebook()
        logger.info("Notebook creation initiated.")

        # Add imports to their own cells
        logger.debug("Adding import cells.")
        nb.cells.append(nbf.v4.new_code_cell("""
        import seaborn as sns
        import matplotlib.pyplot as plt
        from collections import Counter
        import networkx as nx
        import pandas as pd
        """))
        
        
       

        # Add a markdown cell for the introduction
        logger.debug("Adding markdown introduction.")
        nb.cells.append(nbf.v4.new_markdown_cell("# User Analysis"))

        # Add code for defining common properties as a variable in its own cell
        logger.debug("Adding common properties cell.")
        nb.cells.append(nbf.v4.new_markdown_cell("### Most Common Properties"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        common_properties = {common_properties}
        common_properties
        """))

        # Add code for visualizing the most common properties
        logger.debug("Adding visualization for common properties.")
        nb.cells.append(nbf.v4.new_markdown_cell("#### Visualization of Most Common Properties"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        for prop, values in common_properties.items():
            labels, counts = zip(*values)  # Unpack values into labels and counts
            plt.figure(figsize=(12, 6))
            sns.barplot(x=labels, y=counts, palette='viridis')
            plt.title(f'Most Common {{prop.capitalize()}}')
            plt.xlabel(f'{{prop.capitalize()}}')
            plt.ylabel('Count')
            plt.xticks(rotation=90)  # Rotate and align labels
            plt.tight_layout()  # Adjust layout to prevent clipping
            plt.show()
        """))

        logger.info("Visualization of most common properties added.")

        # After loading data and processing common properties, call generate_pie_charts
        logger.debug("Adding pie chart generation.")
        nb.cells.append(nbf.v4.new_markdown_cell("#### Visualization of Distribution of user records in Pie Charts"))

        nb.cells.append(nbf.v4.new_code_cell("""
        def plot_pie_chart(data, title):
            # Count occurrences of each category
            counter = Counter(data)
            labels = counter.keys()
            sizes = counter.values()

            # Plot pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3', len(labels)))
            plt.title(title)
            plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
            plt.show()

        def generate_pie_charts(gender_data, employment_data, subscription_data, address_data):
            # Plot pie charts for each attribute
            plot_pie_chart(gender_data, 'Gender Distribution')
            plot_pie_chart(employment_data, 'Employment Distribution')
            plot_pie_chart(subscription_data, 'Subscription Distribution')
            plot_pie_chart(address_data, 'Address Distribution')
        """))

        # Prepare data for each category
        logger.debug("Preparing data for gender, employment, subscription, and address.")
        users = self.session.query(User).all()

        # Collect data for each category
        gender_data = [user.gender for user in users]
        employment_data = [self.simplify_employment(user.employment) for user in users]
        subscription_data = [self.simplify_subscription(user.subscription) for user in users]
        address_data = [self.simplify_address(user.address) for user in users]

        # Apply the filter: Keep only categories with at least 3 occurrences in the original list
        employment_counter = Counter(employment_data)
        employment_data_filtered = [item for item in employment_data if employment_counter[item] >= 3]
        logger.info(f"Filtered employment data (at least 3 occurrences per category). Total filtered: {len(employment_data_filtered)}")

        # Add the data to the notebook for visualization
        logger.debug("Adding data and calling pie chart generation.")
        nb.cells.append(nbf.v4.new_markdown_cell("### Distribution of Gender, Employment, Subscription, and Address"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        gender_data = {gender_data}
        employment_data = {employment_data_filtered}
        employment_data_full = {employment_data}
        subscription_data = {subscription_data}
        address_data = {address_data}
        """))

        nb.cells.append(nbf.v4.new_code_cell("""
        generate_pie_charts(gender_data, employment_data, subscription_data, address_data)
        """))

        logger.debug("Pie chart generation cell added.")

        # Similarity Analysis - Gender and Employment
        logger.debug("Adding similarity analysis section for Gender and Employment.")
        nb.cells.append(nbf.v4.new_markdown_cell("### Similarity between users based on gender and employment"))
        nb.cells.append(nbf.v4.new_code_cell("""
        def plot_similarity_heatmap(gender_data, employment_data):
            data = pd.DataFrame({
                "Gender": gender_data,
                "Employment": employment_data
            })
            similarity_matrix = pd.crosstab(data["Gender"], data["Employment"])

            plt.figure(figsize=(10, 6))
            sns.heatmap(similarity_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
            plt.title("Similarity Heatmap: Gender and Employment")
            plt.xlabel("Employment")
            plt.ylabel("Gender")
            plt.tight_layout()
            plt.show()
        """))

        nb.cells.append(nbf.v4.new_code_cell("""
        plot_similarity_heatmap(gender_data, employment_data_full)
        """))

        logger.debug("Similarity heatmap function added.")

        # Generate user similarity notebook
        logger.debug("Generating similarity analysis notebook.")
        self.generate_similarity_notebook(users, self.session, nb)

        # Graph Visualizations
        logger.debug("Adding graph visualizations for user similarities.")
        nb.cells.append(nbf.v4.new_markdown_cell("### User Similarities Network Graphs"))

        # Gender and Employment Graph
        logger.debug("Adding graph for Gender and Employment similarities.")
        nb.cells.append(nbf.v4.new_markdown_cell("### Gender and Employment Similarities"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        G_gender_employment = {nx.node_link_data(G_gender_employment)}
        G_gender_employment = nx.node_link_graph(G_gender_employment)

        threshold = 0.5
        strong_edges = [(u, v) for u, v, data in G_gender_employment.edges(data=True) if data['weight'] > threshold]

        G_strong = G_gender_employment.edge_subgraph(strong_edges).copy()
        plt.figure(figsize=(12, 10))
        pos = nx.kamada_kawai_layout(G_strong)
        nx.draw(G_strong, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G_strong, 'weight')
        nx.draw_networkx_edge_labels(G_strong, pos, edge_labels=edge_labels, font_size=9)
        plt.title('Gender and Employment Similarities Network (Strong Edges Only)')
        plt.show()
        """))
        
        # Gender and Employment Graph
        logger.debug("Adding graph for G_gender_subscription ")
        nb.cells.append(nbf.v4.new_markdown_cell("### Gender and Subscription Similarities"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        G_gender_subscription = {nx.node_link_data(G_gender_subscription)}
        G_gender_subscription = nx.node_link_graph(G_gender_subscription)

        threshold = 0.5
        strong_edges = [(u, v) for u, v, data in G_gender_subscription.edges(data=True) if data['weight'] > threshold]

        G_strong = G_gender_subscription.edge_subgraph(strong_edges).copy()
        plt.figure(figsize=(12, 10))
        pos = nx.kamada_kawai_layout(G_strong)
        nx.draw(G_strong, pos, with_labels=True, node_size=300, node_color='orange', font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G_strong, 'weight')
        nx.draw_networkx_edge_labels(G_strong, pos, edge_labels=edge_labels, font_size=9)
        plt.title('Gender and Subscription Similarities Network (Strong Edges Only)')
        plt.show()
        """))
        
        # Gender and Subscription Graph
        logger.debug("Adding graph for G_address_subscription ")
        nb.cells.append(nbf.v4.new_markdown_cell("### Gender and Subscription Similarities"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        G_address_subscription = {nx.node_link_data(G_address_subscription)}
        G_address_subscription = nx.node_link_graph(G_address_subscription)

        threshold = 0.5
        strong_edges = [(u, v) for u, v, data in G_address_subscription.edges(data=True) if data['weight'] > threshold]

        G_strong = G_address_subscription.edge_subgraph(strong_edges).copy()
        plt.figure(figsize=(12, 10))
        pos = nx.kamada_kawai_layout(G_strong)
        nx.draw(G_strong, pos, with_labels=True, node_size=300, node_color='indigo', font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G_strong, 'weight')
        nx.draw_networkx_edge_labels(G_strong, pos, edge_labels=edge_labels, font_size=9)
        plt.title('Gender and Subscription Similarities Network (Strong Edges Only)')
        plt.show()
        """))
        
        
        # Gender and Subscription Graph
        logger.debug("Adding graph for G_address_gender_employment ")
        nb.cells.append(nbf.v4.new_markdown_cell("### Gender and Subscription Similarities"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        G_address_gender_employment = {nx.node_link_data(G_address_gender_employment)}
        G_address_gender_employment = nx.node_link_graph(G_address_gender_employment)

        threshold = 0.5
        strong_edges = [(u, v) for u, v, data in G_address_gender_employment.edges(data=True) if data['weight'] > threshold]

        G_strong = G_address_gender_employment.edge_subgraph(strong_edges).copy()
        plt.figure(figsize=(12, 10))
        pos = nx.kamada_kawai_layout(G_strong)
        nx.draw(G_strong, pos, with_labels=True, node_size=300, node_color='indigo', font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G_strong, 'weight')
        nx.draw_networkx_edge_labels(G_strong, pos, edge_labels=edge_labels, font_size=9)
        plt.title('Gender and Subscription Similarities Network (Strong Edges Only)')
        plt.show()
        """))

        logger.info("Notebook Generation Completed.")

        # Save the notebook
        notebook_path = self.output_file
        try:
            with open(notebook_path, 'w') as f:
                nbf.write(nb, f)
            logger.info(f"Analysis successfully saved to {notebook_path}")
        except Exception as e:
            logger.error(f"Error writing the notebook: {e}")
# Main function for analysis
def main():
    logger.info("Starting the user analysis...")
    analysis = UserAnalysis()

    # Step 1: Create the table and load data
    analysis.create_table()
    
    # Step 2: Load the data into the database
    analysis.load_data()

    # Step 3: Find the most common properties
    common_properties = analysis.find_most_common_properties()
    
    
    

    # Step 4: Find similarities between users
    G_gender_employment, G_address_subscription, G_gender_subscription, G_address_gender_employment = analysis.find_similarities_between_users()

    # Step 5: Generate the Jupyter notebook with the results
    analysis.create_notebook(common_properties, 
                                    G_gender_employment, 
                                    G_address_subscription, 
                                    G_gender_subscription, 
                                    G_address_gender_employment)

    # Close the session
    analysis.close_session()

if __name__ == '__main__':
    main()
