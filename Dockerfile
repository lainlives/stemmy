FROM nvidia/cuda:12.8.1-runtime-rockylinux9

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860


RUN dnf -y install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm
RUN dnf config-manager --set-enabled crb 
RUN dnf update -y
RUN dnf install -y python3.12 python3.12-devel ffmpeg git python3-pip python3.12-pip 
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir /app
RUN cd /app && git clone https://github.com/lainlives/stemmy

WORKDIR /app/stemmy


RUN cd /app/stemmy && python3.12 -m pip install --no-cache-dir --upgrade -r requirements.txt
RUN mkdir /tmp/models
RUN python3.12 -c "from assets.model_tools import download_file; download_file('https://huggingface.co/lainlives/audio-separator-models/raw/main/assets/luvr5-ui/models.txt', '/tmp')"
RUN python3.12 -c "from assets.model_tools import download_files_from_txt; download_files_from_txt('/tmp/models.txt', '/tmp/models/')"


# 7. Expose the port (HF Spaces defaults to 7860)
EXPOSE 7860

# 8. Start the application (Example using Gradio or FastAPI)
CMD ["python3.12", "webui.py"]
