# ---- 1.  base image with conda pre-installed ----
FROM continuumio/miniconda3:latest

# ---- 2.  copy only the env spec first (Docker layer caching) ----
WORKDIR /app
COPY streamlit_app/environment.yml .

# ---- 3.  create the conda env *exactly* as you do locally ----
RUN conda env create -f environment.yml -n vehicle_detection && \
    conda clean -afy

# ---- 4.  make sure the env is active for every RUN / CMD ----
ENV PATH=/opt/conda/envs/vehicle_detection/bin:$PATH

# ---- 5.  copy the rest of the application code ----
COPY streamlit_app/ ./

# ---- 6.  expose Streamlit’s default port ----
EXPOSE 8501

# ---- 7.  launch the app ----
CMD ["streamlit", "run", "app_yolo.py", "--server.port=8501", "--server.address=0.0.0.0"]
