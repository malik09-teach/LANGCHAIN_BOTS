from crewai import Agent
from tools import load_csv_data

def get_data_engineer():
    return Agent(
        role='Senior Data Engineer',
        goal='Analyze raw dataset structures and provide concrete preprocessing steps.',
        backstory='You are a master of pandas, scikit-learn, and data cleaning. You excel at spotting missing values, unscaled numerical variables, and categorical features that need encoding.',
        tools=[load_csv_data],
        verbose=True
    )

def get_model_architect():
    return Agent(
        role='Deep Learning Architect',
        goal='Design optimal neural network code based on clean dataset characteristics.',
        backstory='You are an expert machine learning engineer specializing in TensorFlow and PyTorch. You look at data shapes and types to design high-performance model architectures.',
        verbose=True
    )