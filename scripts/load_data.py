import nbformat as nbf
import pandas as pd
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.schema import Base, User
from fuzzywuzzy import fuzz

# Ensure the 'logs' directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

log_file_path = os.path.join('logs', 'app.log')

# Set up the logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to the console
        logging.FileHandler(log_file_path)  # Output to a log file inside the 'logs' folder
    ]
)

logger = logging.getLogger(__name__)

# SQLAlchemy Engine and Session setup
DATABASE_URL = "sqlite:///users.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

class UserAnalysis:
    def __init__(self, csv_file='data/random_users.csv'):
        self.csv_file = csv_file
        self.session = Session()

    def create_table(self):
        """Create the database table using SQLAlchemy models."""
        User.__table__.drop(engine)  # Drop the existing table
        logger.info("Dropped existing table.")
        Base.metadata.create_all(engine)
        logger.info("Table created (if not already exists).")

    def load_data(self):
        """Load data from the CSV file into the database."""
        logger.info(f"Loading data from {self.csv_file}")
        df = pd.read_csv(self.csv_file)
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


    def find_similar_users(self):
        """Find similar users based on matching gender, username, or other properties."""
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

                if gender_similarity or username_similarity > 80:
                    similar_pairs.append((user1.username, user2.username, username_similarity, gender_similarity))

        logger.info(f"Found {len(similar_pairs)} similar user pairs.")
        return similar_pairs

    def close_session(self):
        """Close the SQLAlchemy session."""
        self.session.close()
        logger.info("Database session closed.")

# Main function for analysis
def main():
    logger.info("Starting the user analysis...")
    analysis = UserAnalysis()

    # Step 1: Create the table and load data
    analysis.create_table()
    analysis.load_data()

    # Step 2: Find similar users
    similar_users = analysis.find_similar_users()

    # Step 3: Prepare DataFrame for display
    similar_users_df = pd.DataFrame(similar_users, columns=["User 1", "User 2", "Username Similarity", "Gender Match"])
    logger.debug(f"DataFrame Head: {similar_users_df.head()}")

    # Step 4: Write analysis to the notebook
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# User Similarity Analysis"))
    nb.cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt"))
    
    # Check if the DataFrame was created successfully
    print(similar_users_df.head())

    # Add the DataFrame to the notebook
    # nb.cells.append(nbf.v4.new_code_cell(f"similar_users_df = {similar_users_df.to_dict(orient='records')}"))
    nb.cells.append(nbf.v4.new_code_cell(f"similar_users_df = pd.DataFrame({similar_users}, columns=['User 1', 'User 2', 'Username Similarity', 'Gender Match'])"))
    nb.cells.append(nbf.v4.new_markdown_cell("### Data preview"))
    nb.cells.append(nbf.v4.new_code_cell("similar_users_df.head()"))

    # Visualization of Username Similarity
    nb.cells.append(nbf.v4.new_markdown_cell("### Distribution of Username Similarity"))
    nb.cells.append(nbf.v4.new_code_cell("""
    plt.figure(figsize=(10, 6))
    sns.histplot(similar_users_df['Username Similarity'], kde=True, color='blue', bins=20)
    plt.title('Distribution of Username Similarity Scores')
    plt.xlabel('Username Similarity')
    plt.ylabel('Frequency')
    plt.show()
    """))


    # Gender Match Visualization
    nb.cells.append(nbf.v4.new_markdown_cell("### Gender Match Distribution"))
    nb.cells.append(nbf.v4.new_code_cell("""
    gender_match_counts = similar_users_df['Gender Match'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=gender_match_counts.index, y=gender_match_counts.values, palette='viridis')
    plt.title('Gender Match Distribution')
    plt.xlabel('Gender Match (1=Match, 0=No Match)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No Match', 'Match'])
    plt.show()
    """))

    # Save the notebook
    notebook_path = "notebooks/analysis.ipynb"
    try:
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)
        logger.info(f"Analysis successfully saved to {notebook_path}")
    except Exception as e:
        logger.error(f"Error writing the notebook: {e}")

    # Close the session
    analysis.close_session()

# Ensure the script runs directly
if __name__ == "__main__":
    main()
