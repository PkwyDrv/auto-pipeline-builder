from setuptools import setup, find_packages

setup(
    name="auto_pipeline",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain>=0.0.300",
        "openai>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0,<2.1.0",
        "scikit-learn>=1.3.0",
        "feature-engine>=1.6.0",
        "sweetviz>=2.2.0",
        "ydata-profiling<4.16.0",
        "mlflow>=2.8.0",
        "prefect>=2.13.0",
        "gradio>=4.0.0",
        "great-expectations<0.18.0",
    ],
    python_requires=">=3.8",
) 