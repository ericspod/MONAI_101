FROM python:3.12-slim

# Set the working (current) directory
WORKDIR /app

# Set environment variables, this isn't really necessary for this example
ENV PYTHONPATH=/app

# Copy project files from current directory into /app, this will include source files and the weight file plus other 
# things we don't really need. The .dockerignore file lists things this command will ignore when copying.
COPY . /app

# Upgrade pip itself and install requirements. This is done as a single RUN command with && joining the commands.
# This is done to avoid making multiple layers. Docker creates new layers for every command which take up space.
# Adding --no-cache-dir to pip calls ensures the download module archives aren't taking up space in the image itself.
RUN pip install --upgrade --no-cache-dir pip && pip install --no-cache-dir -r requirements.txt

# Run the application, it's assumed /inputs is a mount point the user provides
CMD ["python", "monai_101_infer.py", "/inputs"]
