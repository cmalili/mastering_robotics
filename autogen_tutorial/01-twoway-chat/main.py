from autogen_core.components import config

def main():
    config_list = config.config_list_from_json(
        env_or_file = "OAI_CONFIG_LIST.json",
    )