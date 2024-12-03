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
from fuzzywuzzy import fuzz
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

            logger.info(f"Most common properties: {common_properties}")
            return common_properties
        except Exception as e:
            logger.error(f"Error during finding common properties: {e}")

    # Helper methods to simplify complex fields
    def simplify_employment(self, employment_data):
        # Extract the key skill or title from employment (you can choose which value you want to keep)
        # Assuming employment_data is a string representing a dictionary
        if isinstance(employment_data, str):
            employment_data = eval(employment_data)
        return employment_data.get('title', '')

    def simplify_subscription(self, subscription_data):
        # Extract the plan or the status from subscription
        if isinstance(subscription_data, str):
            subscription_data = eval(subscription_data)
        return subscription_data.get('plan', '')

    def simplify_address(self, address_data):
        # Extract the city or street name from the address
        if isinstance(address_data, str):
            address_data = eval(address_data)
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
        
        for i in range(len(properties[0])):  # Assuming all lists are of equal length
            for j in range(i + 1, len(properties[0])):
                similarity_score = similarities[i][j]
                if similarity_score > 0.5:  # Define a threshold for a strong connection
                    G.add_edge(i, j, weight=similarity_score, type='strong')
                elif similarity_score > 0.1:  # Define a threshold for a weak connection
                    G.add_edge(i, j, weight=similarity_score, type='weak')

        return G
    
    # def find_similar_users(self):
    #     """Find similar users based on matching gender, username, or other properties."""
    #     users = self.session.query(User).all()
    #     similar_pairs = []

    #     for i in range(len(users)):
    #         for j in range(i + 1, len(users)):
    #             user1 = users[i]
    #             user2 = users[j]

    #             # Compare gender similarity
    #             gender_similarity = 1 if user1.gender == user2.gender else 0

    #             # Compare username similarity using fuzzy matching
    #             username_similarity = fuzz.ratio(user1.username, user2.username)

    #             if gender_similarity or username_similarity > 80:
    #                 similar_pairs.append((user1.username, user2.username, username_similarity, gender_similarity))

    #     logger.info(f"Found {len(similar_pairs)} similar user pairs.")
    #     return similar_pairs
    def find_similar_users(self):
        """
        Find similar users based on matching gender, username, employment, or address properties.
        """
        users = self.session.query(User).all()
        similar_pairs = []

        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1 = users[i]
                user2 = users[j]

                # Compare gender similarity
                gender_similarity = 1 if user1.gender == user2.gender else 0

                # Compare username similarity using fuzzy matching
                username_similarity = fuzz.ratio(user1.username, user2.username)
                
                employ1 = self.simplify_employment(user1.employment)
                employ2 = self.simplify_employment(user2.employment)
                # Compare employment similarity (exact match or fuzzy matching for job titles)
                employment_similarity = 1 if employ1 == employ2 else fuzz.ratio(employ1, employ2)
                
                sub1 = self.simplify_subscription(user1.subscription)
                sub2 = self.simplify_subscription(user2.subscription)
                # Compare subscription similarity (exact match or fuzzy matching for subscription plans)
                subscription_similarity = 1 if sub1 == sub2 else fuzz.ratio(sub1, sub2)

                address1 = self.simplify_address(user1.address)
                address2 = self.simplify_address(user2.address)
                
                # Compare address similarity using fuzzy matching
                address_similarity = fuzz.ratio(address1, address2)

                # Define thresholds for username, employment, and address similarities
                if (
                    gender_similarity
                    or username_similarity > 80
                    or employment_similarity > 80
                    or address_similarity > 80
                    or subscription_similarity > 80
                ):
                    similar_pairs.append(
                        (
                            user1.username,
                            user2.username,
                            username_similarity,
                            gender_similarity,
                            employment_similarity,
                            address_similarity,
                            subscription_similarity
                        )
                    )

        logger.info(f"Found {len(similar_pairs)} similar user pairs.")
        return similar_pairs

    
    def close_session(self):
        """Close the SQLAlchemy session."""
        self.session.close()
        logger.info("Database session closed.")
        
           
    
    # Function to generate notebook
    def create_notebook(self, common_properties, G_gender_employment, G_address_subscription, G_gender_subscription, G_address_gender_employment):
        # Create a new notebook
        nb = nbf.v4.new_notebook()

        # Add imports to their own cells
        nb.cells.append(nbf.v4.new_code_cell("""
        import seaborn as sns
        import matplotlib.pyplot as plt
        from collections import Counter
        import networkx as nx
        import pandas as pd

        """))

        # Add a markdown cell for the introduction
        nb.cells.append(nbf.v4.new_markdown_cell("# User Analysis"))

        # Add code for defining common properties as a variable in its own cell
        nb.cells.append(nbf.v4.new_markdown_cell("### Most Common Properties"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        common_properties = {common_properties}
        common_properties
        """))

        # Add code for visualizing the most common properties
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
        
        # After loading data and processing common properties, call generate_pie_charts
        # self.generate_pie_charts()
        nb.cells.append(nbf.v4.new_markdown_cell("#### Visualization of  Distribution of user records in Pie Charts"))

                # Add the pie chart function directly in the notebook
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
        
        users = self.session.query(User).all()

        # Prepare data for each category
        gender_data = [user.gender for user in users]
        employment_data = [self.simplify_employment(user.employment) for user in users]
        subscription_data = [self.simplify_subscription(user.subscription) for user in users]
        address_data = [self.simplify_address(user.address) for user in users]


        # Apply the filter: Keep only categories with at least 3 occurrences in the original list
        employment_counter = Counter(employment_data)
        employment_data_filtered = [item for item in employment_data if employment_counter[item] >= 3]
        
        nb.cells.append(nbf.v4.new_markdown_cell("### Distribution of Gender, Employment, Subscription, and Address"))


        nb.cells.append(nbf.v4.new_code_cell(f"""
        # Define the data variables (replace with actual data if needed)
        gender_data = {gender_data}
        employment_data = {employment_data_filtered}
        employment_data_full = {employment_data}
        subscription_data ={subscription_data}
        address_data = {address_data}
        """))
        
        # Add code to define data variables directly in the notebook
        nb.cells.append(nbf.v4.new_code_cell(f"""
    
        # Call the function to generate the pie charts
        generate_pie_charts(gender_data, employment_data, subscription_data, address_data)
        """))
        
         # Add code to define data variables directly in the notebook
        nb.cells.append(nbf.v4.new_markdown_cell("### similarity between users based on gender and employment"))
        nb.cells.append(nbf.v4.new_code_cell("""
        
        # Define the function to plot the similarity heatmap
        def plot_similarity_heatmap(gender_data, employment_data):
            # Create a DataFrame
            data = pd.DataFrame({
                "Gender": gender_data,
                "Employment": employment_data
            })

            # Create a cross-tabulation of Gender and Employment
            similarity_matrix = pd.crosstab(data["Gender"], data["Employment"])

            # Plot the heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(similarity_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
            plt.title("Similarity Heatmap: Gender and Employment")
            plt.xlabel("Employment")
            plt.ylabel("Gender")
            plt.tight_layout()
            plt.show()
        
        """))
        
        # Add code to define data variables directly in the notebook
        nb.cells.append(nbf.v4.new_code_cell(f"""
    
        # Call the function to generate the relationship heatmap
        plot_similarity_heatmap(gender_data, employment_data_full)
        """))
        
        nb.cells.append(nbf.v4.new_markdown_cell("### similarity between users using fuzzy match"))

        # Step 2: Find similar users
        similar_users = self.find_similar_users()

        # Step 3: Prepare DataFrame for display
        similar_users_df = pd.DataFrame(
            similar_users, 
            columns=[
                "User 1", 
                "User 2", 
                "Username Similarity", 
                "Gender Match", 
                "Employment Similarity", 
                "Address Similarity", 
                "Subscription Similarity"
            ]
        )
        logger.debug(f"DataFrame Head: {similar_users_df.head()}")
   

        # Add the DataFrame to the notebook
        nb.cells.append(
        nbf.v4.new_code_cell(
        f"""
        similar_users_df = pd.DataFrame(
                    {similar_users}, 
                    columns=[
                        "User 1", 
                        "User 2", 
                        "Username Similarity", 
                        "Gender Match", 
                        "Employment Similarity", 
                        "Address Similarity", 
                        "Subscription Similarity"
                    ]
                )               
        """
        )
        )
        nb.cells.append(nbf.v4.new_markdown_cell("### Data preview"))
        nb.cells.append(nbf.v4.new_code_cell("similar_users_df.head()"))

        # Visualization of Username Similarity
        nb.cells.append(nbf.v4.new_markdown_cell("### Distribution of Username Similarity in fuzzy matching"))
        nb.cells.append(nbf.v4.new_code_cell("""
        plt.figure(figsize=(10, 6))
        sns.histplot(similar_users_df['Username Similarity'], kde=True, color='blue', bins=20)
        plt.title('Distribution of Username Similarity Scores')
        plt.xlabel('Username Similarity')
        plt.ylabel('Frequency')
        plt.show()
        """))
        
          # Visualization of Employment Similarity
        nb.cells.append(nbf.v4.new_markdown_cell("### Distribution of Employment Similarity in fuzzy matching"))
        nb.cells.append(nbf.v4.new_code_cell("""
        plt.figure(figsize=(10, 6))
        sns.histplot(similar_users_df['Employment Similarity'], kde=True, color='blue', bins=20)
        plt.title('Distribution of Employment Similarity Scores')
        plt.xlabel('Employment Similarity')
        plt.ylabel('Frequency')
        plt.show()
        """))
        
        
          # Visualization of Subscription Similarity
        nb.cells.append(nbf.v4.new_markdown_cell("### Distribution of Subscription Similarity in fuzzy matching"))
        nb.cells.append(nbf.v4.new_code_cell("""
        plt.figure(figsize=(10, 6))
        sns.histplot(similar_users_df['Employment Similarity'], kde=True, color='blue', bins=20)
        plt.title('Distribution of Subscription Similarity Scores')
        plt.xlabel('Subscription Similarity')
        plt.ylabel('Frequency')
        plt.show()
        """))
        
          # Visualization of Address Similarity
        nb.cells.append(nbf.v4.new_markdown_cell("### Distribution of Address Similarity in fuzzy matching"))
        nb.cells.append(nbf.v4.new_code_cell("""
        plt.figure(figsize=(10, 6))
        sns.histplot(similar_users_df['Address Similarity'], kde=True, color='blue', bins=20)
        plt.title('Distribution of Address Similarity Scores')
        plt.xlabel('Address Similarity')
        plt.ylabel('Frequency')
        plt.show()
        """))
            
            



        
        # Add code for similarity network visualization
        nb.cells.append(nbf.v4.new_markdown_cell("### User Similarities Network Graphs"))
         # Add code for similarity network visualizations for each graph
        # Visualize the graph with adjusted layout and filtered edges
        nb.cells.append(nbf.v4.new_markdown_cell("### Gender and Employment Similarities"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        G_gender_employment = {nx.node_link_data(G_gender_employment)}
        G_gender_employment = nx.node_link_graph(G_gender_employment)

        # Set a threshold for strong edges (e.g., weight > 0.5)
        threshold = 0.5
        strong_edges = [(u, v) for u, v, data in G_gender_employment.edges(data=True) if data['weight'] > threshold]

        # Create a subgraph with only strong edges
        G_strong = G_gender_employment.edge_subgraph(strong_edges).copy()

        # Visualize the graph with Kamada-Kawai layout and only strong edges
        plt.figure(figsize=(12, 10))
        pos = nx.kamada_kawai_layout(G_strong)  # Using Kamada-Kawai layout for better clarity in dense graphs
        nx.draw(G_strong, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G_strong, 'weight')
        nx.draw_networkx_edge_labels(G_strong, pos, edge_labels=edge_labels, font_size=9)
        plt.title('Gender and Employment Similarities Network (Strong Edges Only)')
        plt.show()
        """))

        # Address and Subscription Similarities Network (Strong Edges Only)
        nb.cells.append(nbf.v4.new_markdown_cell("### Address and Subscription Similarities (Strong Edges Only)"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        G_address_subscription = {nx.node_link_data(G_address_subscription)}
        G_address_subscription = nx.node_link_graph(G_address_subscription)

        # Set a threshold for strong edges (e.g., weight > 0.5)
        threshold = 0.5
        strong_edges = [(u, v) for u, v, data in G_address_subscription.edges(data=True) if data['weight'] > threshold]

        # Create a subgraph with only strong edges
        G_strong = G_address_subscription.edge_subgraph(strong_edges).copy()

        # Visualize the graph with Kamada-Kawai layout and only strong edges
        plt.figure(figsize=(10, 8))
        pos = nx.kamada_kawai_layout(G_strong)  # Using Kamada-Kawai layout
        nx.draw(G_strong, pos, with_labels=True, node_size=500, node_color='lightgreen', font_size=10, font_weight='bold')

        # Get edge labels
        edge_labels = nx.get_edge_attributes(G_strong, 'weight')

        # Draw edge labels
        nx.draw_networkx_edge_labels(G_strong, pos, edge_labels=edge_labels)

        plt.title('Address and Subscription Similarities Network (Strong Edges Only)')
        plt.show()
        """))

        # Gender and Subscription Similarities Network (Strong Edges Only)
        nb.cells.append(nbf.v4.new_markdown_cell("### Gender and Subscription Similarities (Strong Edges Only)"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        G_gender_subscription = {nx.node_link_data(G_gender_subscription)}
        G_gender_subscription = nx.node_link_graph(G_gender_subscription)

        # Set a threshold for strong edges (e.g., weight > 0.5)
        threshold = 0.5
        strong_edges = [(u, v) for u, v, data in G_gender_subscription.edges(data=True) if data['weight'] > threshold]

        # Create a subgraph with only strong edges
        G_strong = G_gender_subscription.edge_subgraph(strong_edges).copy()

        # Visualize the graph with Kamada-Kawai layout and only strong edges
        plt.figure(figsize=(10, 8))
        pos = nx.kamada_kawai_layout(G_strong)  # Using Kamada-Kawai layout
        nx.draw(G_strong, pos, with_labels=True, node_size=500, node_color='lightcoral', font_size=10, font_weight='bold')

        # Get edge labels
        edge_labels = nx.get_edge_attributes(G_strong, 'weight')

        # Draw edge labels
        nx.draw_networkx_edge_labels(G_strong, pos, edge_labels=edge_labels)

        plt.title('Gender and Subscription Similarities Network (Strong Edges Only)')
        plt.show()
        """))

        # Address, Gender, and Employment Similarities Network (Strong Edges Only)
        nb.cells.append(nbf.v4.new_markdown_cell("### Address, Gender, and Employment Similarities (Strong Edges Only)"))
        nb.cells.append(nbf.v4.new_code_cell(f"""
        G_address_gender_employment = {nx.node_link_data(G_address_gender_employment)}
        G_address_gender_employment = nx.node_link_graph(G_address_gender_employment)

        # Set a threshold for strong edges (e.g., weight > 0.5)
        threshold = 0.5
        strong_edges = [(u, v) for u, v, data in G_address_gender_employment.edges(data=True) if data['weight'] > threshold]

        # Create a subgraph with only strong edges
        G_strong = G_address_gender_employment.edge_subgraph(strong_edges).copy()

        # Visualize the graph with Kamada-Kawai layout and only strong edges
        plt.figure(figsize=(10, 8))
        pos = nx.kamada_kawai_layout(G_strong)  # Using Kamada-Kawai layout
        nx.draw(G_strong, pos, with_labels=True, node_size=500, node_color='lightskyblue', font_size=10, font_weight='bold')

        # Get edge labels
        edge_labels = nx.get_edge_attributes(G_strong, 'weight')

        # Draw edge labels
        nx.draw_networkx_edge_labels(G_strong, pos, edge_labels=edge_labels)

        plt.title('Address, Gender, and Employment Similarities Network (Strong Edges Only)')
        plt.show()
        """))


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
