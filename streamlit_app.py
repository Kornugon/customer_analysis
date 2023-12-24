import streamlit as st
import torch

import plotly.express as px
import pandas as pd
import os
# import warnings
# warnings.filterwarnings('ignore')

#  streamlit run streamlit_app.py

from streamlit_app.app_config_update import ConfigUpdater


markdown_in_variable = "####"
markdown_in_exp_variable = "#####"

st.set_page_config(
    page_title="Customer Analysis",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.pl_itp',  # TODO
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


st.title(" :bar_chart: Customer Analysis Project")
st.markdown(
    "<style>div.block-container{padding-top:1rem;}</style>",
    unsafe_allow_html=True
)

fl = st.file_uploader(
    ":file_folder: Upload a file",
    type=(["csv", "txt", "xlsx", "xls"])
)


with st.sidebar:
    end_model, end_task = None, None
    st.header("Choose model & task:")
    model_name = st.radio(
        f"{markdown_in_variable} Choose Model",
        ("RNN", "Transformer"),
        index=0,
        horizontal=True
    )
    selected_task = st.radio(
        f"{markdown_in_variable} Choose task",
        ("Churn analysis", "Next event prediction"),
        index=0,
        horizontal=True,
        help="""
        Whether to predict user churn or next event.
        """
    )
    task = "churn" if selected_task == "Churn analysis" else "events"

    if end_model != model_name or end_task != task:
        if 'lambda_' in st.session_state:
            del st.session_state['lambda_']
        if "lr_" in st.session_state:
            del st.session_state['lr_']

    config_updater = ConfigUpdater(model_name, task)
    config_updater.update_section("model_training_params", task=task)

    st.header("Set model training configuration:")
    with st.expander("Model initial parameters:", expanded=False):
        config_section = "model_init_params"

        hidden_size = int(st.number_input(
            f"{markdown_in_exp_variable} hidden size",
            value=config_updater.get_param(config_section, "hidden_size"),
            step=2,
            min_value=32,
            help="""
            The number of features in the hidden state.
            """
        ))
        update_params = {"hidden_size": hidden_size}

        num_layers = int(st.number_input(
            f"{markdown_in_exp_variable} number of layers",
            value=config_updater.get_param(config_section, "num_layers"),
            step=2,
            min_value=2,
            help="""
            Number of recurrent layers.
            """
        ))
        update_params["num_layers"] = num_layers

        num_heads = int(st.number_input(
            f"{markdown_in_exp_variable} number of heads",
            value=config_updater.get_param(config_section, "num_heads"),
            step=1,
            min_value=1,
            help="""
            Number of heads for multi-head attention.
            Only used, when attention_type is 'multi-head'.
            """
        ))
        update_params["num_heads"] = num_heads

        config_updater.update_section(config_section, **update_params)
        config_updater.save()

    with st.expander("Model training parameters:", expanded=False):
        config_section = "model_training_params"

        device_options = ("cpu", "cuda")
        device = st.radio(
            f"{markdown_in_exp_variable} Choose compute device",
            options=device_options,
            index=device_options.index(
                config_updater.get_param(config_section, "device")),
            horizontal=True,
            disabled=not torch.cuda.is_available(),
            help="""
            Computation device. If a GPU is not available,
            the radio button will be disabled and 'cpu' will be used.
            """
        )
        update_params = {"device": device}

        num_epochs = int(st.number_input(
            f"{markdown_in_exp_variable} number of epochs",
            value=config_updater.get_param(config_section, "num_epochs"),
            step=1,
            min_value=1,
            help="""
            Number of training epochs.
            """
        ))
        update_params["num_epochs"] = num_epochs

        batch_size = int(st.number_input(
            f"{markdown_in_exp_variable} batch size",
            value=config_updater.get_param(config_section, "batch_size"),
            step=4,
            min_value=4,
            help="""
            Amount of samples per batch.
            """
        ))
        update_params["batch_size"] = batch_size

        num_workers = int(st.number_input(
            f"{markdown_in_exp_variable} number of workers",
            value=config_updater.get_param(config_section, "num_workers"),
            step=2,
            min_value=0,
            help="""
            How many subprocesses to use for data loading.
            '0' means, that the data will be loaded in the main process.
            """
        ))
        update_params["num_workers"] = num_workers

        config_updater.update_section(config_section, **update_params)
        config_updater.save()

    with st.expander("Pipeline parameters:", expanded=False):
        config_section = "pipeline_params"

        shuffle_train_dataloader = bool(st.checkbox(
            f"{markdown_in_exp_variable} shuffle train dataloader data",
            value=config_updater.get_param(
                config_section, "shuffle_train_dataloader"),
            help="""
            Whether to shuffle training data (train dataloader)
            before training or not.
            """
        ))
        update_params = {"shuffle_train_dataloader": shuffle_train_dataloader}

        eval_model = bool(st.checkbox(
            f"{markdown_in_exp_variable} evaluate model",
            value=config_updater.get_param(config_section, "eval_model"),
            help="""
            Whether to count other scores (classification metrics) along with
            'grid search metric' or not.
            """
        ))
        update_params["eval_model"] = eval_model

        save_attention_weights = bool(st.checkbox(
            f"{markdown_in_exp_variable} save attention weights",
            value=config_updater.get_param(
                config_section, "save_attention_weights"),
            help="""
            Whether to save attention weights or not.
            """
        ))
        update_params["save_attention_weights"] = save_attention_weights

        return_churn_prob = bool(st.checkbox(
            f"{markdown_in_exp_variable} return churn probabilities",
            value=config_updater.get_param(
                config_section, "return_churn_prob"),
            help="""
            Whether to return churn probabilities or boolean value.
            """
        ))
        update_params["return_churn_prob"] = return_churn_prob

        prob_thresold = float(st.number_input(
            f"{markdown_in_exp_variable} churn probability thresold",
            value=config_updater.get_param(config_section, "prob_thresold"),
            step=0.01,
            min_value=0.1,
            max_value=0.95,
            disabled=not return_churn_prob,
            help="""
            Churn probability threshold above which, boolean value will
            be True otherwise False.
            """
        ))
        update_params["prob_thresold"] = prob_thresold

        early_stopping_patience = int(st.number_input(
            f"{markdown_in_exp_variable} early stopping patience",
            value=config_updater.get_param(
                config_section, "early_stopping_patience"),
            step=1,
            min_value=3,
            help="""
            Number of epochs without improvement (patience) of validation
            loss during training. For example, if set to 3, then after
            3 epochs without improvement, training will be stopped.
            """
        ))
        update_params["early_stopping_patience"] = early_stopping_patience

        metric_options = ["accuracy", "precision", "recall", "f1"]
        grid_search_metric = str(st.selectbox(
            f"{markdown_in_exp_variable} grid search metric",
            options=metric_options,
            index=metric_options.index(config_updater.get_param(
                config_section, "grid_search_metric")),
            help="""
            Grid search metric to determine the best mode hyperparameters set,
            based on the chosen metric score.
            """
        ))
        update_params["grid_search_metric"] = grid_search_metric

        config_updater.update_section(config_section, **update_params)
        config_updater.save()

    with st.expander("Grid search parameters:", expanded=False):
        config_section = "grid_search_params"

        nonlinearity_options = ["relu", "tanh"]
        nonlinearity = st.multiselect(
            f"{markdown_in_exp_variable} non-linearity",
            options=nonlinearity_options,
            default=config_updater.get_param(
                config_section, "nonlinearity"),
            max_selections=len(nonlinearity_options),
            help="""
            The non-linearity to use by the RNN model.
            """
        )
        attention_options = ["multi-head", "global", "self"]
        attention_type = st.multiselect(
            f"{markdown_in_exp_variable} attention type",
            options=attention_options,
            default=config_updater.get_param(
                config_section, "attention_type"),
            max_selections=len(attention_options),
            help="""
            Type of attention mechanism to use by the RNN model.
            """
        )
        if "lr_" not in st.session_state:
            st.session_state.lr_ = config_updater.get_param(
                config_section, "learning_rate")
        lr_label_ = "add learning rate value for 'learning rate'"
        lr_ = float(st.number_input(
            label=f"{markdown_in_exp_variable} {lr_label_}",
            value=st.session_state.lr_[0],
            step=0.001,
            min_value=0.000000000001,
            format="%f",
            help="""
            Add lambda value for below multiselect option.
            """
        )
        )
        if lr_ and lr_ not in st.session_state.lr_:
            st.session_state.lr_.append(lr_)
        learning_rate = st.multiselect(
            f"{markdown_in_exp_variable} learning rate",
            options=st.session_state.lr_,
            default=None,
            max_selections=len(st.session_state.lr_),
            help="""
            Learning rate to use in the pipeline optimizer.
            """
        )
        regularization_options = ["L1", "L2"]
        reg_type = st.multiselect(
            f"{markdown_in_exp_variable} regularization type",
            options=regularization_options,
            default=config_updater.get_param(config_section, "reg_type"),
            max_selections=len(regularization_options),
            help="""
            Regularization type to use in the pipeline.
            """
        )
        if reg_type:
            if "lambda_" not in st.session_state:
                st.session_state.lambda_ = config_updater.get_param(
                    config_section, "reg_lambda")
            label_ = "add lambda value for 'regularization lambda'"
            lambda_ = float(st.number_input(
                label=f"{markdown_in_exp_variable} {label_}",
                value=st.session_state.lambda_[0],
                step=0.0001,
                min_value=0.000000000001,
                format="%f",
                help="""
                Add lambda value for below multiselect option.
                """
            )
            )
            if lambda_ and lambda_ not in st.session_state.lambda_:
                st.session_state.lambda_.append(lambda_)

            reg_lambda = st.multiselect(
                f"{markdown_in_exp_variable} regularization lambda",
                options=st.session_state.lambda_,
                default=config_updater.get_param(
                    config_section, "reg_lambda"),
                max_selections=len(st.session_state.lambda_),
                help="""
                Regularization lambda values to use in the pipeline.
                """
            )
    end_model, end_task = model_name, task
