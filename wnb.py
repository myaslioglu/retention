import wandb

class ExperimentTracker:
    def __init__(self, project_name, config=None):
        self.project_name = project_name
        self.config = config or {}
        self.run = None
    
    def __enter__(self):
        self.run = wandb.init(project=self.project_name, config=self.config)
        return self.run
    
    def __exit__(self, *args):
        wandb.finish()
