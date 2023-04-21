# Base image
FROM python:3.10.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY scraper.py .
COPY ML_Model.py .
COPY DL_Model.py .
COPY app.py .
COPY svm_model.pkl .
COPY svm_vectorizer.pkl .
COPY dl_model.pkl .
COPY dl_tokenizer.pkl .
COPY dl_max_length.pkl .

# Expose the Streamlit port
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
