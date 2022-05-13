

import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath


def get_train_input(TRAIN_SET_COUNT: int, TRAIN_SET_LIMIT: int,  TRAIN_INPUT_JSON: kfp.components.OutputPath(str), TRAIN_OUTPUT_JSON: kfp.components.OutputPath(str)):
    """Generate a random number between minimum and maximum (inclusive)."""
    from random import randint
    import json
    TRAIN_INPUT = list()
    TRAIN_OUTPUT = list()
    for i in range(TRAIN_SET_COUNT):
        a = randint(0, TRAIN_SET_LIMIT)
        b = randint(0, TRAIN_SET_LIMIT)
        c = randint(0, TRAIN_SET_LIMIT)
        op = a + (2 * b) + (3 * c)
        TRAIN_INPUT.append([a, b, c])
        TRAIN_OUTPUT.append(op)
    jsonString = json.dumps(TRAIN_INPUT)
    with open(TRAIN_INPUT_JSON, 'w') as odd_writer:
        odd_writer.write(jsonString)
    jsonString = json.dumps(TRAIN_OUTPUT)
    with open(TRAIN_OUTPUT_JSON, 'w') as odd_writer:
        odd_writer.write(jsonString)

get_train_input_op = kfp.components.func_to_container_op(get_train_input)


def train_evaluation(k: int, TRAIN_INPUT_JSON: str, TRAIN_OUTPUT_JSON: str) -> str:
    """Flip a coin and output heads or tails randomly."""
    import sys
    import subprocess
    j = k
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','sklearn'])

    import sklearn
    from sklearn.linear_model import LinearRegression
    predictor = LinearRegression(n_jobs=-1)
    import json
    predictor.fit(X=json.loads(TRAIN_INPUT_JSON), y=json.loads(TRAIN_OUTPUT_JSON))

    X_TEST = [[10, 20, 30]]
    outcome = predictor.predict(X=X_TEST)
    coefficients = predictor.coef_

    return 'Outcome : {}\nCoefficients : {}'.format(outcome, coefficients)


train_evaluation_op = kfp.components.func_to_container_op(train_evaluation)


@dsl.pipeline(
    name='No merging pipeline',
    description='Shows the cost of running an unmerged pipeline'
)
def halfMerge_pipeline():
    out1 = get_train_input_op(2000, 100)
    #output = train_evaluation_op(-1, out1.outputs['TRAIN_INPUT_JSON'], out1.outputs['TRAIN_OUTPUT_JSON'])


# Submit the pipeline for execution:
# kfp.Client(host=kfp_endpoint).create_run_from_pipeline_func(flipcoin_pipeline, arguments={})
if __name__ == '__main__':
    # Compiling the pipeline

    # pipeline_func = noMerge_pipeline
    # pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    # import kfp.compiler as compiler
    # compiler.Compiler().compile(pipeline_func, pipeline_filename)

    client = kfp.Client()
    # Specify pipeline argument values
    arguments = {}  # whatever makes sense for new version
    # Submit a pipeline run
    client.create_run_from_pipeline_func(halfMerge_pipeline, arguments=arguments)
