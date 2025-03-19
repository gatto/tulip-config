import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class ConfigValidator:
    """Validates configuration files for the Tulip application."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the validator with the configuration directory.
        
        Args:
            config_dir: Path to the configuration directory. If None, tries to detect it.
        """
        if config_dir is None:
            # Try to locate the config directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_config_dirs = [
                # When running from packages/server/src
                os.path.abspath(os.path.join(script_dir, "..", "..", "..", "config")),
                # When running from project root
                os.path.abspath(os.path.join(script_dir, "config")),
            ]
            
            for path in possible_config_dirs:
                if os.path.isdir(path) and os.path.exists(os.path.join(path, "master.yaml")):
                    config_dir = path
                    break
                    
            if config_dir is None:
                raise ConfigValidationError(
                    "Could not locate config directory. Please specify the path explicitly."
                )
        
        self.config_dir = Path(config_dir)
        self.master_config_path = self.config_dir / "master.yaml"
        self.problems_dir = self.config_dir / "problems"
        self.dashboards_dir = self.config_dir / "dashboards"
        
        # Ensure required directories exist
        if not self.config_dir.exists():
            raise ConfigValidationError(f"Config directory does not exist: {self.config_dir}")
        if not self.master_config_path.exists():
            raise ConfigValidationError(f"Master config file not found: {self.master_config_path}")
        if not self.problems_dir.exists():
            raise ConfigValidationError(f"Problems directory not found: {self.problems_dir}")
        if not self.dashboards_dir.exists():
            raise ConfigValidationError(f"Dashboards directory not found: {self.dashboards_dir}")
    
    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and parse a YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Parsed YAML content
            
        Raises:
            ConfigValidationError: If the file cannot be loaded or parsed
        """
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigValidationError(f"Error loading {file_path}: {str(e)}")
    
    def validate_master_config(self) -> Dict[str, Any]:
        """
        Validate the master configuration file.
        
        Returns:
            The validated master configuration
            
        Raises:
            ConfigValidationError: If validation fails
        """
        master_config = self.load_yaml(self.master_config_path)
        
        # Check required sections
        required_sections = ['dashboards', 'problems']
        for section in required_sections:
            if section not in master_config:
                raise ConfigValidationError(f"Master config is missing required section: {section}")
            
            if not isinstance(master_config[section], list):
                raise ConfigValidationError(f"Master config section '{section}' must be a list")
        
        # Validate dashboards
        for i, dashboard in enumerate(master_config.get('dashboards', [])):
            self._validate_dashboard_entry(dashboard, i)
        
        # Validate problems
        for i, problem in enumerate(master_config.get('problems', [])):
            self._validate_problem_entry(problem, i)
        
        # Validate valid_combinations if present
        if 'valid_combinations' in master_config:
            if not isinstance(master_config['valid_combinations'], list):
                raise ConfigValidationError("valid_combinations must be a list")
            
            for i, combo in enumerate(master_config['valid_combinations']):
                if not isinstance(combo, dict):
                    raise ConfigValidationError(f"Combination {i} must be a dictionary")
                
                if 'dashboard' not in combo:
                    raise ConfigValidationError(f"Combination {i} is missing required field: dashboard")
                
                if 'problems' not in combo:
                    raise ConfigValidationError(f"Combination {i} is missing required field: problems")
                
                if not isinstance(combo['problems'], list):
                    raise ConfigValidationError(f"Combination {i} 'problems' field must be a list")
        
        return master_config
    
    def _validate_dashboard_entry(self, dashboard: Dict[str, Any], index: int) -> None:
        """Validate a dashboard entry in the master config."""
        required_fields = ['id', 'name', 'description']
        self._validate_required_fields(dashboard, required_fields, f"Dashboard {index}")
        
        # Validate dashboard id format (simple alphanumeric check)
        if not dashboard['id'].isalnum():
            raise ConfigValidationError(
                f"Dashboard id '{dashboard['id']}' must be alphanumeric"
            )
    
    def _validate_problem_entry(self, problem: Dict[str, Any], index: int) -> None:
        """Validate a problem entry in the master config."""
        required_fields = ['id', 'name', 'description']
        self._validate_required_fields(problem, required_fields, f"Problem {index}")
        
        # Validate problem id format (simple alphanumeric check)
        if not problem['id'].isalnum():
            raise ConfigValidationError(
                f"Problem id '{problem['id']}' must be alphanumeric"
            )
    
    def _validate_required_fields(
        self, item: Dict[str, Any], required_fields: List[str], item_name: str
    ) -> None:
        """Check if all required fields are present in an item."""
        for field in required_fields:
            if field not in item:
                raise ConfigValidationError(f"{item_name} is missing required field: {field}")
            
            if not item[field]:  # Check for empty values
                raise ConfigValidationError(f"{item_name} has empty value for field: {field}")
    
    def validate_dashboard_config(self, dashboard_id: str) -> Dict[str, Any]:
        """
        Validate a dashboard configuration file.
        
        Args:
            dashboard_id: The ID of the dashboard to validate
            
        Returns:
            The validated dashboard configuration
            
        Raises:
            ConfigValidationError: If validation fails
        """
        dashboard_path = self.dashboards_dir / f"{dashboard_id}.yaml"
        if not dashboard_path.exists():
            raise ConfigValidationError(f"Dashboard config not found: {dashboard_path}")
        
        dashboard_config = self.load_yaml(dashboard_path)
        
        # Check for required sections
        if 'dashboard' not in dashboard_config:
            raise ConfigValidationError(f"Dashboard config {dashboard_id} is missing 'dashboard' section")
        
        dashboard = dashboard_config['dashboard']
        required_fields = ['name', 'description', 'layout']
        self._validate_required_fields(dashboard, required_fields, f"Dashboard {dashboard_id}")
        
        # Validate layout
        if not isinstance(dashboard['layout'], dict):
            raise ConfigValidationError(f"Dashboard {dashboard_id} layout must be a dictionary")
        
        # Validate panels if present
        if 'panels' in dashboard_config:
            if not isinstance(dashboard_config['panels'], list):
                raise ConfigValidationError(f"Dashboard {dashboard_id} 'panels' must be a list")
            
            for i, panel in enumerate(dashboard_config['panels']):
                if not isinstance(panel, dict):
                    raise ConfigValidationError(f"Dashboard {dashboard_id} panel {i} must be a dictionary")
                
                required_panel_fields = ['id', 'title', 'type']
                self._validate_required_fields(panel, required_panel_fields, f"Panel {i} in dashboard {dashboard_id}")
        
        return dashboard_config
    
    def validate_problem_config(self, problem_id: str) -> Dict[str, Any]:
        """
        Validate a problem configuration file.
        
        Args:
            problem_id: The ID of the problem to validate
            
        Returns:
            The validated problem configuration
            
        Raises:
            ConfigValidationError: If validation fails
        """
        problem_path = self.problems_dir / f"{problem_id}.yaml"
        if not problem_path.exists():
            raise ConfigValidationError(f"Problem config not found: {problem_path}")
        
        problem_config = self.load_yaml(problem_path)
        
        # Check for required sections
        if 'problem' not in problem_config:
            raise ConfigValidationError(f"Problem config {problem_id} is missing 'problem' section")
        
        problem = problem_config['problem']
        required_fields = ['name', 'description']
        self._validate_required_fields(problem, required_fields, f"Problem {problem_id}")
        
        # Validate features if present
        if 'features' in problem_config:
            if not isinstance(problem_config['features'], dict):
                raise ConfigValidationError(f"Problem {problem_id} 'features' must be a dictionary")
            
            for feature_name, feature in problem_config['features'].items():
                if not isinstance(feature, dict):
                    raise ConfigValidationError(
                        f"Problem {problem_id} feature '{feature_name}' must be a dictionary"
                    )
                
                if 'type' not in feature:
                    raise ConfigValidationError(
                        f"Problem {problem_id} feature '{feature_name}' is missing 'type'"
                    )
        
        # Validate model if present
        if 'model' in problem_config:
            if not isinstance(problem_config['model'], dict):
                raise ConfigValidationError(f"Problem {problem_id} 'model' must be a dictionary")
            
            model = problem_config['model']
            if 'type' not in model:
                raise ConfigValidationError(f"Problem {problem_id} model is missing 'type'")
        
        return problem_config
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Validate all configuration files.
        
        Returns:
            A dictionary with validation results
            
        Raises:
            ConfigValidationError: If validation fails
        """
        results = {
            'master': None,
            'dashboards': {},
            'problems': {},
            'errors': []
        }
        
        # Validate master config
        try:
            master_config = self.validate_master_config()
            results['master'] = master_config
        except ConfigValidationError as e:
            results['errors'].append(f"Master config error: {str(e)}")
            return results
        
        # Validate all dashboards
        for dashboard in master_config.get('dashboards', []):
            dashboard_id = dashboard['id']
            try:
                dashboard_config = self.validate_dashboard_config(dashboard_id)
                results['dashboards'][dashboard_id] = dashboard_config
            except ConfigValidationError as e:
                results['errors'].append(f"Dashboard '{dashboard_id}' error: {str(e)}")
        
        # Validate all problems
        for problem in master_config.get('problems', []):
            problem_id = problem['id']
            try:
                problem_config = self.validate_problem_config(problem_id)
                results['problems'][problem_id] = problem_config
            except ConfigValidationError as e:
                results['errors'].append(f"Problem '{problem_id}' error: {str(e)}")
        
        return results


def print_validation_report(results: Dict[str, Any]) -> None:
    """Print a human-readable validation report."""
    if results['errors']:
        print("❌ Configuration validation failed with the following errors:")
        for error in results['errors']:
            print(f"  - {error}")
        return
    
    print("✅ All configuration files are valid!")
    print(f"  - Master config: OK")
    print(f"  - Dashboards: {len(results['dashboards'])} valid")
    print(f"  - Problems: {len(results['problems'])} valid")


def main():
    """Run the validator as a standalone tool."""
    # Allow specifying config directory from command line
    config_dir = None
    if len(sys.argv) > 1:
        config_dir = sys.argv[1]
    
    try:
        validator = ConfigValidator(config_dir)
        results = validator.validate_all()
        print_validation_report(results)
        
        # Exit with error code if there were errors
        if results['errors']:
            sys.exit(1)
            
    except ConfigValidationError as e:
        print(f"❌ Configuration validation error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
