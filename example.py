# Other imports
import numpy as np
import torch
import pandas as pd

# DeepCASE Imports
from deepcase.preprocessing   import Preprocessor
from deepcase.context_builder import ContextBuilder
from deepcase.interpreter     import Interpreter

if __name__ == "__main__":
    preprocessor = Preprocessor(length  = 10, timeout = 86400)
    context, events, labels, mapping = preprocessor.csv('save/alerts.csv', verbose=False)
    context_builder = ContextBuilder.load('save/builder.save')
    interpreter = Interpreter.load('save/interpreter.save', context_builder)
    unique_test = torch.load('save/context_test.pt')
    unique_train = torch.load('save/context_train.pt')

    if torch.cuda.is_available():
        events  = events .to('cuda')
        context = context.to('cuda')
        context_builder = context_builder.to('cuda')

    events_train  = events [:events.shape[0]//5 ]
    events_test   = events [ events.shape[0]//5:]

    context_train = context[:events.shape[0]//5 ]
    context_test  = context[ events.shape[0]//5:]

    labels_train  = labels [:events.shape[0]//5 ]
    labels_test   = labels [ events.shape[0]//5:]

    pick = range(20)
    # print(f"context={context_train[unique_train][pick]}")
    # print(f"events={events_train[unique_train][pick]}")

    clusters = interpreter.cluster(context_train[unique_train][pick], events_train[unique_train][pick].reshape(-1, 1))
    cluster_indices = np.where(clusters == 0)[0]
    print(f"{clusters=}")
    print(f"{cluster_indices=}")

    scores = interpreter.score_clusters(labels_train[unique_train][pick])
    interpreter.score(scores)
    prediction = interpreter.predict(context_test[unique_test][pick], events_test[unique_test][pick].reshape(-1, 1))
