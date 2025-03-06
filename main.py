from utils_data import load_train_data, setup_env, load_metadata, assert_feature_shape
from utils_training import run_cross_val
from models import choose_model
import click



@click.command()
@click.option("--config-path", default="./config.yaml", help="Path to the configuration file.")
def main(config_path):
    # Load Config and create folder environment
    config, data_dir, save_dir, repeat = setup_env(config_path)

    # Load Metadata
    df_train, df_test, train_features_dir, test_features_dir = load_metadata(config, data_dir)

    # Load Data
    X, y = load_train_data(df_train, train_features_dir)
    print("X and y shape=", X.shape, y.shape)

    assert_feature_shape(X, config.get("features"))

    # Get Training Parameters (from config)
    training_params = {key: config[key] for key in ["batch_size", "nb_epochs", "learning_rate", "upsample"] if key in config}

    # Select Model according to Config
    Model, model_params = choose_model(config)

    # Run the Cross Validation
    df_eval, models_cv = run_cross_val(X, y, df_train, save_dir, Model, model_params, repeat, training_params)
    print(df_eval)

if __name__ == "__main__":
    main()
