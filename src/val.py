from factorymlops.validators.YOLOValidator import YOLOValidator
from datetime import datetime as dt
from ruamel.yaml import YAML
import os


def save_val_results(metrics, model_name: str, output_path: str):
    timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
    path = '{}_eval_{}.yaml'.format(model_name, timestamp)

    os.makedirs(output_path, exist_ok=True)

    output_yaml_path = os.path.join(output_path, path)

    yaml = YAML()

    with open(output_yaml_path, 'w') as file:
        yaml.dump(metrics, file)

    return output_yaml_path


if __name__ == "__main__":
    dataset_yaml_path = "dataset.yaml"
    model_path = str(input("Chemin du modèle"))
    output_path = "../eval_results"
    model_name = str(input("Nom du modèle"))

    validator = YOLOValidator()
    metrics = validator.eval(model_path, dataset_yaml_path)

    output_yaml_path = save_val_results(metrics, model_name, output_path)

    print("Résultat de l'eval sauvegardé dans : ", output_yaml_path)
