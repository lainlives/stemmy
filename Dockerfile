FROM nvidia/cuda:12.8.1-runtime-rockylinux9
# 2. Set environment variables to prevent interactive prompts and set the HF port
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# 3. Install system dependencies (Python, Git, etc.)
RUN dnf update && dnf install -y python3.13  git  && rm -rf /var/lib/apt/lists/*


RUN cd $HOME/app && git clone https://github.com/lainlives/stemmy

WORKDIR $HOME/app/stemmy

# 5. Copy and install Python requirements
# Ensure you include 'torch' with the correct CUDA version in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 6. Copy the rest of your application code
COPY --chown=user . .

# 7. Expose the port (HF Spaces defaults to 7860)
EXPOSE 7860

# 8. Start the application (Example using Gradio or FastAPI)
CMD ["python3", "app.py"]
