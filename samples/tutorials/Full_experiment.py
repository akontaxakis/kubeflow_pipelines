
import kfp
from kfp import dsl


def A(TRAIN_SET_COUNT: int, TRAIN_SET_LIMIT: int,  TRAIN_INPUT_JSON: kfp.components.OutputPath(str)):
    from random import randint
    import json
    TRAIN_INPUT = list()
    for i in range(TRAIN_SET_COUNT):
        a = randint(0, TRAIN_SET_LIMIT)
        b = randint(0, TRAIN_SET_LIMIT)
        c = randint(0, TRAIN_SET_LIMIT)
        TRAIN_INPUT.append([a, b, c])
    jsonString = json.dumps(TRAIN_INPUT)
    with open(TRAIN_INPUT_JSON, 'w') as odd_writer:
        odd_writer.write(jsonString)


A_op = kfp.components.func_to_container_op(A)

def B(TRAIN_SET_COUNT: int, TRAIN_INPUT_JSON: kfp.components.InputPath(str),TRAIN_OUTPUT_JSON: kfp.components.OutputPath(str)):
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

B_op = kfp.components.func_to_container_op(B)


def AB(TRAIN_SET_COUNT: int, TRAIN_SET_LIMIT: int,  TRAIN_INPUT_JSON: kfp.components.OutputPath(str), TRAIN_OUTPUT_JSON: kfp.components.OutputPath(str)):
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

AB_op = kfp.components.func_to_container_op(AB)

def ABC(TRAIN_SET_COUNT: int, TRAIN_SET_LIMIT: int, testD: kfp.components.OutputPath(str)):
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


    import sys
    import subprocess

    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'sklearn'])

    import sklearn
    from sklearn.linear_model import LinearRegression

    predictor = LinearRegression(n_jobs=-1)
    import json

    predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

    X_TEST = [[10, 20, 30]]
    outcome = predictor.predict(X=X_TEST)
    coefficients = predictor.coef_
    with open(testD, 'w') as odd_writer:
        odd_writer.write('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))



ABC_op = kfp.components.func_to_container_op(ABC)

def C(TRAIN_INPUT_JSON: str, TRAIN_OUTPUT_JSON: str, testD: kfp.components.OutputPath(str)):
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
    with open(testD, 'w') as odd_writer:
        odd_writer.write('Outcome! : {}\nCoefficients : {}'.format(outcome, coefficients))


C_op = kfp.components.func_to_container_op(C)

def D(testD: kfp.components.InputPath(str)):
    """Flip a coin and output heads or tails randomly."""
    print("!!!"+testD)


D_op = kfp.components.func_to_container_op(D)

def ABCD(TRAIN_SET_COUNT: int, TRAIN_SET_LIMIT: int) ->str:
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


    import sys
    import subprocess

    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'sklearn'])

    import sklearn
    from sklearn.linear_model import LinearRegression

    predictor = LinearRegression(n_jobs=-1)
    import json

    predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

    X_TEST = [[10, 20, 30]]
    outcome = predictor.predict(X=X_TEST)
    coefficients = predictor.coef_

    return '!!Outcome : {}\nCoefficients : {}'.format(outcome, coefficients)

ABCD_op = kfp.components.func_to_container_op(ABCD)

@dsl.pipeline(
    name='No merging pipeline',
    description='Shows the cost of running an unmerged pipeline'
)
def noMerge_pipeline():
    out1 = A_op(1000, 1000)
    out2 = B_op(1000, out1.outputs['TRAIN_INPUT_JSON'])
    out3 = C_op(out1.outputs['TRAIN_INPUT_JSON'], out2.outputs['TRAIN_OUTPUT_JSON'])
    D_op(out3.outputs['testD'])


def halfMerge_pipeline():
    out3 = ABC_op(1000, 1000)
    D_op(out3.outputs['testD'])

def fullMerge_pipeline():
    out3 = ABCD_op(1000, 1000)



if __name__ == '__main__':

    client = kfp.Client()
    arguments = {}  # whatever makes sense for new version

    client.create_run_from_pipeline_func(fullMerge_pipeline, arguments=arguments)
