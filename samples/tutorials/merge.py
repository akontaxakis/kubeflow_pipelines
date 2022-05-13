

import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath


def get_train_input(TRAIN_SET_COUNT: int, TRAIN_SET_LIMIT: int):
    """Generate a random number between minimum and maximum (inclusive)."""
    from random import randint
    import json
    TRAIN_SET_LIMIT = 1000
    TRAIN_SET_COUNT = 100
    import sys
    import subprocess

    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'sklearn'])
    TRAIN_INPUT = list()
    TRAIN_OUTPUT = list()
    print(type(TRAIN_OUTPUT))
    for i in range(TRAIN_SET_COUNT):
        a = randint(0, TRAIN_SET_LIMIT)
        b = randint(0, TRAIN_SET_LIMIT)
        c = randint(0, TRAIN_SET_LIMIT)
        op = a + (2 * b) + (3 * c)
        TRAIN_INPUT.append([a, b, c])
        TRAIN_OUTPUT.append(op)

    jsonString = json.dumps(TRAIN_INPUT)
    print(jsonString)

    TRAIN_INPUT = json.loads(jsonString)
    print(type(jsonString))
    for i in range(TRAIN_SET_COUNT):
        print(TRAIN_INPUT)
    print(TRAIN_OUTPUT)
    from sklearn.linear_model import LinearRegression

    predictor = LinearRegression(n_jobs=-1)
    predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

    print(type(predictor))
    X_TEST = [[10, 20, 30]]
    outcome = predictor.predict(X=X_TEST)
    coefficients = predictor.coef_

    print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))


get_train_input_op = kfp.components.func_to_container_op(get_train_input)


def get_train_output(TRAIN_SET_COUNT: int, TRAIN_INPUT_JSON: kfp.components.InputPath(str),TRAIN_OUTPUT_JSON: kfp.components.OutputPath(str)):
    """Generate a random number between minimum and maximum (inclusive)."""
    from random import randint
    import json
    TRAIN_OUTPUT = list()
    with open(TRAIN_INPUT_JSON, 'r') as reader:
        TRAIN_INPUT = json.loads(reader.readline())
    for i in range(TRAIN_SET_COUNT):
        a = TRAIN_INPUT.__getitem__(i)[0]
        b = TRAIN_INPUT.__getitem__(i)[1]
        c = TRAIN_INPUT.__getitem__(i)[2]
        op = a + (2 * b) + (3 * c)
        TRAIN_OUTPUT.append(op)
    jsonString = json.dumps(TRAIN_OUTPUT)
    with open(TRAIN_OUTPUT_JSON, 'w') as odd_writer:
        odd_writer.write(jsonString)



get_train_output_op = kfp.components.func_to_container_op(get_train_output)


def train_evaluation(TRAIN_INPUT_JSON: str, TRAIN_OUTPUT_JSON: str) -> str:
    """Flip a coin and output heads or tails randomly."""
    import sys
    import subprocess

    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'sklearn'])

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
def Merge_pipeline():
    out1 = get_train_input_op(1000, 10000)
    #out2 = get_train_output_op(1000, out1.outputs['TRAIN_INPUT_JSON'])
    #output = train_evaluation_op(out1.outputs['TRAIN_INPUT_JSON'], out2.outputs['TRAIN_OUTPUT_JSON'])


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
    client.create_run_from_pipeline_func(Merge_pipeline, arguments=arguments)
