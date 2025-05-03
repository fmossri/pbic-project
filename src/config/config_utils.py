from .models import AppConfig

def check_config_changes(current_config: AppConfig, new_config: AppConfig):
    update_fields = []
        
    # Iterate through field names
    for field_name in new_config.model_fields.keys():
        # Use getattr for safe attribute access
        current_value = getattr(current_config, field_name)
        new_value = getattr(new_config, field_name)
        if current_value != new_value:
            update_fields.append(field_name) # Append the name

    return update_fields