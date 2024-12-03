import requests
import pandas as pd
import time
import os
import logging

# Ensure the 'logs' directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

log_file_path = os.path.join('logs', 'app.log')

# Set up the logger
logging.basicConfig(
    level=logging.INFO,  # You can adjust the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to the console
        logging.FileHandler(log_file_path)  # Output to a log file inside the 'logs' folder
    ]
)

# Get the logger
logger = logging.getLogger(__name__)


class RandomUserFetcher:
    def __init__(self, num_users=1000, batch_size=100, output_file='data/random_users.csv'):
        self.num_users = num_users
        self.batch_size = batch_size
        self.output_file = output_file
        self.base_url = "https://random-data-api.com/api/v2/users"
        self.all_users = []

        # Ensure the data directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
    
    def fetch_users(self):
        """Fetch random users in batches and store them incrementally."""
        for i in range(0, self.num_users, self.batch_size):
            remaining = self.num_users - len(self.all_users)
            current_batch_size = min(self.batch_size, remaining)
            
            logger.info(f"Fetching {current_batch_size} users...")
            
            try:
                response = requests.get(f"{self.base_url}?size={current_batch_size}")
                if response.status_code == 200:
                    data = response.json()
                    self.all_users.extend(data)
                    # print(f"Fetched {len(data)} users (Total: {len(self.all_users)} users)")
                    logger.info(f"Fetched {len(data)} users (Total: {len(self.all_users)} users)")

                    # Progressive saving to CSV
                    self.save_to_csv(progress=True)
                else:
                    # print(f"Error fetching data: {response.status_code}")
                    logger.error(f"Error fetching data: {response.status_code}")

                    break
            except Exception as e:
                # print(f"Exception occurred: {e}")
                logger.error(f"Exception occurred: {e}")  # Replaced print with logger

            
            # Delay to respect API rate limits
            time.sleep(1)

            if len(self.all_users) >= self.num_users:
                break

        # print(f"Finished fetching. Total users fetched: {len(self.all_users)}")
        logger.info(f"Finished fetching. Total users fetched: {len(self.all_users)}")  # Replaced print with logger


    def save_to_csv(self, progress=False):
        """Save the fetched users to a CSV file."""
        df = pd.DataFrame(self.all_users)
        file_name = self.output_file if not progress else self.output_file.replace('.csv', '_progress.csv')
        df.to_csv(file_name, index=False)
        if progress:
            # print(f"Progress saved to {file_name}")
            logger.info(f"Progress saved to {file_name}")  # Replaced print with logger

        else:
            # print(f"Saved {len(self.all_users)} users to {file_name}")
            logger.info(f"Saved {len(self.all_users)} users to {file_name}")  # Replaced print with logger


if __name__ == "__main__":
    fetcher = RandomUserFetcher(num_users=1000, batch_size=100)
    fetcher.fetch_users()
    fetcher.save_to_csv()
