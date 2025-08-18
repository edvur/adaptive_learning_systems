"""
config.py - Central Configuration Management
Provides portable paths and settings for the adaptive tutoring system
"""

from pathlib import Path
from typing import Dict
import os

class Config:
    """Central configuration for the adaptive tutoring system"""
    
    def __init__(self):
        # Base directories - portable across systems
        self.BASE_DIR = Path(__file__).parent.parent  # Points to project root
        self.SRC_DIR = Path(__file__).parent          # Points to src/
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.SRC_DIR / "models"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.FIGURES_DIR = self.RESULTS_DIR / "figures"
        self.METRICS_DIR = self.RESULTS_DIR / "metrics"
        self.ASSETS_DIR = self.SRC_DIR / "assets"
        
        # Model-specific paths
        self.MODEL_PATHS = {
            'feature_pipeline': self.MODELS_DIR / "feature_pipeline.pkl",
            'Perception': self.MODELS_DIR / "final_model_Perception.pkl",
            'Input': self.MODELS_DIR / "final_model_Input.pkl", 
            'Understanding': self.MODELS_DIR / "final_model_Understanding.pkl",
            'dqn_model': self.MODELS_DIR / "best_model.pth"
        }
        
        # Data file paths
        self.DATA_PATHS = {
            'csms': self.DATA_DIR / "CSMS.xlsx",
            'cshs': self.DATA_DIR / "CSHS.xlsx"
        }
        
        # Database path
        self.DATABASE_PATH = self.SRC_DIR / "study_data.db"
        
        # Training configuration
        self.RANDOM_STATE = 42
        
        # Create necessary directories
        self.create_directories()
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR, 
            self.RESULTS_DIR,
            self.FIGURES_DIR,
            self.METRICS_DIR,
            self.ASSETS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self, data_type: str) -> Path:
        """Get path for specific data file"""
        if data_type in self.DATA_PATHS:
            return self.DATA_PATHS[data_type]
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def get_model_path(self, model_type: str) -> Path:
        """Get path for specific model file"""
        if model_type in self.MODEL_PATHS:
            return self.MODEL_PATHS[model_type]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all critical paths exist"""
        validation_results = {}
        
        # Check data files
        for name, path in self.DATA_PATHS.items():
            validation_results[f"data_{name}"] = path.exists()
        
        # Check model files
        for name, path in self.MODEL_PATHS.items():
            validation_results[f"model_{name}"] = path.exists()
        
        return validation_results
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path from project root"""
        return self.BASE_DIR / relative_path
    
    def __str__(self):
        """String representation of configuration"""
        return f"""
Adaptive Tutoring System Configuration:
- Base Directory: {self.BASE_DIR}
- Data Directory: {self.DATA_DIR}
- Models Directory: {self.MODELS_DIR}
- Results Directory: {self.RESULTS_DIR}
- Database: {self.DATABASE_PATH}
"""

# Global configuration instance
CONFIG = Config()

# Convenience functions
def get_config() -> Config:
    """Get the global configuration instance"""
    return CONFIG

def get_data_path(data_type: str) -> Path:
    """Convenience function to get data path"""
    return CONFIG.get_data_path(data_type)

def get_model_path(model_type: str) -> Path:
    """Convenience function to get model path"""
    return CONFIG.get_model_path(model_type)

if __name__ == "__main__":
    # Test configuration
    config = Config()
    print(config)
    
    # Validate paths
    validation = config.validate_paths()
    print("\nPath Validation:")
    for path_name, exists in validation.items():
        status = "OK" if exists else "MISSING"
        print(f"  {status}: {path_name}")