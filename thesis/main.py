from experiments.experiment import SCP_Experiment
from utils import utils

# model configs
resnet1d_wang = {
    "modelname": "resnet1d_wang",
    "parameters": dict(),
}

xresnet1d101 = {
    "modelname": "xresnet1d101",
    "parameters": dict(),
}

inception1d = {
    "modelname": "inception1d",
    "parameters": dict(),
}

lstm = {
    "modelname": "lstm",
    "parameters": dict(lr=1e-3),
}


def main():
    datafolder = "ptbxl/"
    outputfolder = "output/"

    models = [
        #lstm,
        inception1d,
        xresnet1d101,
        resnet1d_wang,
    ]

    experiments = [
        ("exp1.1", "subdiagnostic"),
        ("exp1.1.1", "superdiagnostic"),
    ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        e.perform()
        # e.evaluate()


if __name__ == "__main__":
    main()
