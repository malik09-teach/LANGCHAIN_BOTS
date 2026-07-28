from crewai import Task

def get_analysis_task(agent, file_path):
    return Task(
        description=f"Load and inspect the dataset located at: '{file_path}'. Identify missing values, check if numerical features require scaling, and output a detailed data-cleaning blueprint.",
        expected_output="A markdown report detailing missing data strategies, categorical columns to encode, and columns that must be scaled.",
        agent=agent
    )

def get_modeling_task(agent):
    return Task(
        description="Review the data cleaning blueprint from the previous step. Write clean Python code using standard ML libraries to build an appropriate Neural Network architecture matching this data schema.",
        expected_output="A complete, runnable Python script wrapped in markdown code blocks defining the model preparation and architecture setup.",
        agent=agent
    )