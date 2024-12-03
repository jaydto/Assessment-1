# Define the Python executable
PYTHON = python

# Define scripts paths
FETCH_USERS_SCRIPT = scripts/fetch_users.py
LOAD_DATA_SCRIPT = -m scripts.load_data

# Define targets
.PHONY: fetch_users load_data run_analysis clean

# Target to fetch and save random users
fetch_users:
	$(PYTHON) $(FETCH_USERS_SCRIPT)

# Target to load data into SQLite and perform analysis
load_data:
	$(PYTHON) $(LOAD_DATA_SCRIPT)

run_notebook:
	jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb

run_notebook_view:
	`jupyter notebook notebooks/analysis.ipynb

# Run the entire workflow
run_analysis: fetch_users load_data
	@echo "Workflow completed successfully!"

# Clean up intermediate files or outputs
clean:
	rm -rf logs
	> data/random_users.csv
	> notebooks/analysis.ipynb
	@echo "Cleaned up logs and notebooks and csv!"