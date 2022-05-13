import json

import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath


def get_train_input(TRAIN_SET_COUNT: int, TRAIN_SET_LIMIT: int) -> str:
    """Generate a random number between minimum and maximum (inclusive)."""
    from random import randint
    TRAIN_INPUT = list()
    for i in range(TRAIN_SET_COUNT):
        a = randint(0, TRAIN_SET_LIMIT)
        b = randint(0, TRAIN_SET_LIMIT)
        c = randint(0, TRAIN_SET_LIMIT)
        TRAIN_INPUT.append([a, b, c])
    jsonString = json.dumps(TRAIN_INPUT)
    return '' + jsonString


get_train_input_op = kfp.components.func_to_container_op(get_train_input)


def get_train_output(TRAIN_SET_COUNT: int, TRAIN_INPUT_JSON: str) -> str:
    """Generate a random number between minimum and maximum (inclusive)."""
    from random import randint

    TRAIN_OUTPUT = list()
    TRAIN_INPUT = json.loads(TRAIN_INPUT_JSON)
    for i in range(TRAIN_SET_COUNT):
        a = TRAIN_INPUT.__getitem__(i)[0]
        b = TRAIN_INPUT.__getitem__(i)[1]
        c = TRAIN_INPUT.__getitem__(i)[2]
        op = a + (2 * b) + (3 * c)
        TRAIN_OUTPUT.append(op)

    jsonString = json.dumps(TRAIN_OUTPUT)
    return '' + jsonString


get_train_output_op = kfp.components.func_to_container_op(get_train_output)


def train_evaluation(TRAIN_INPUT_JSON: str, TRAIN_OUTPUT_JSON: str) -> str:
    """Flip a coin and output heads or tails randomly."""
    from sklearn.linear_model import LinearRegression
    predictor = LinearRegression(n_jobs=-1)
    predictor.fit(X=json.loads(TRAIN_INPUT_JSON), y=json.loads(TRAIN_OUTPUT_JSON))

    X_TEST = [[10, 20, 30]]
    outcome = predictor.predict(X=X_TEST)
    coefficients = predictor.coef_

    return 'Outcome : {}\nCoefficients : {}'.format(outcome, coefficients)


train_evaluation_op = kfp.components.func_to_container_op(train_evaluation)


def print_fun(TRAIN_OUTPUT_JSON: str):
    """Flip a coin and output heads or tails randomly."""
    print(str)


print_fun_op = kfp.components.func_to_container_op(print_fun)


@dsl.pipeline(
    name='No merging pipeline',
    description='Shows the cost of running an unmerged pipeline'
)
def noMerge_pipeline():
    out1 = get_train_input_op(100, 1000)
    out2 = get_train_output_op(100, out1.outputs['TRAIN_INPUT_JSON'])
    output = train_evaluation_op(out1.outputs['TRAIN_INPUT_JSON'], out2.outputs['TRAIN_OUTPUT_JSON'])
    print_fun_op(output)

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
    client.create_run_from_pipeline_func(noMerge_pipeline, arguments=arguments)
