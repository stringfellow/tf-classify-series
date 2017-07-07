#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Classify BMS (Building Management System) time series that should be
dependant on Time, OAT (Outside Air Temp) & Schedule.

In theory, given the time of day, OAT and a schedule, some property of the
other TS should respond to this. In the training sets, my assumption is that
they are not faulty, or alternatively, that this will classify also-faulty TS.

Further classification could then be done from faulty and non-faulty TS, where
faulty means things like "is cooling when it's cold outside". For classifying
things like the Control Temp, we can probably assume no faults (maybe the
sensor is a bit dodgy but should roughly respond in some way to OAT)

Data input to the RNN needs to look like this, then:

training sequence = [
    [
        1497139203,  # epoch seconds
        0,           # schedule 'off'
        15,          # OAT
        25           # time-series of interest to classify (CT in this example)
    ],
    ...
    [
        1497190547,
        1,
        21,
        20
    ]
    ...
]

After: http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/

"""
import argparse
import logging
from datetime import datetime
from random import shuffle

import numpy as np
import pandas as pd
import tensorflow as tf


log = logging.getLogger(__name__)


# One-hot encoding of our TS types
TS_CLASSES = {
#    'ct': [0, 0, 0],
    'htg': [0, 0],
    'clg': [0, 1],
}


def get_inputs_and_outputs_for_ts(ts_df, oat_df, type_str):
    """Prepare a single TS with its output class."""
    log.info("Preparing %s", type_str)
    input_outputs = []
    for col in ts_df.columns:
        sequence_grouped = []
        for index, datum in ts_df[col].items():
            sequence_grouped.append([
                (index - datetime(1970, 1, 1)).total_seconds(),  # epoch secs
                # schedule here,
                oat_df[index],
                datum
            ])
        input_outputs.append((
            np.array(sequence_grouped),
            TS_CLASSES[type_str]
        ))
    return input_outputs


def prepare_data():
    """Prepare data into corresponding inputs and outputs."""
    df_ct = pd.read_csv(
        'data/ts/ONC-Office-FCU-RtnTemp.csv_OUT.csv',
        index_col=0,
        parse_dates=True,
    )
    df_htg = pd.read_csv(
        'data/ts/ONC-Office-FCU-HtgVlvSig.csv_OUT.csv',
        index_col=0,
        parse_dates=True,
    )
    df_clg = pd.read_csv(
        'data/ts/ONC-Office-FCU-ClgVlvSig.csv_OUT.csv',
        index_col=0,
        parse_dates=True,
    )
    df_oat = pd.read_csv(
        'data/ts/ONC-Office-OAT.csv_OUT.csv',
        index_col=0,
        parse_dates=True,
    )

    train_input = []
    train_output = []

    oat_col = df_oat.columns[0]
    oat_df = df_oat[oat_col]
    input_outputs = []
#    input_outputs.extend(get_inputs_and_outputs_for_ts(df_ct, oat_df, 'ct'))
    input_outputs.extend(get_inputs_and_outputs_for_ts(df_htg, oat_df, 'htg'))
    input_outputs.extend(get_inputs_and_outputs_for_ts(df_clg, oat_df, 'clg'))

    shuffle(input_outputs)
    for i, o in input_outputs:
        train_input.append(i)
        train_output.append(o)
    return len(input_outputs), train_input, train_output


def get_model(sequence_length, hidden_layers):
    """Generate and return the model.

    I don't really understand quite what is happening beyond setting up the
    LSTMCell...

    """
    with tf.name_scope('input'):
        # 3 dimensions - epoch, OAT and actual value to classify
        data = tf.placeholder(tf.float32, [None, sequence_length, 3])
        # 2 classes - heating and cooling
        target = tf.placeholder(tf.float32, [None, 2])

    cell = tf.contrib.rnn.LSTMCell(hidden_layers, state_is_tuple=True)
    val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    # this is the bit I don't really get - why are we switching batch and
    # sequence size? We just take the last value... which is presumably the
    # class it has finally derived?
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    with tf.name_scope("weight"):
        weight = tf.Variable(tf.truncated_normal([
            hidden_layers, int(target.get_shape()[1])
        ]))

    with tf.name_scope("bias"):
        bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    with tf.name_scope('cross_entropy'):
        cross_entropy = -tf.reduce_sum(
            target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0))
        )

    with tf.name_scope('train'):
        minimize = tf.train.AdamOptimizer().minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    # create a summary for our cost and accuracy
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("error", error)

    # merge all summaries into a single "operation" which we can execute in a
    # session
    summary_op = tf.summary.merge_all()

    model = {
        'minimize': minimize,
        'data': data,
        'target': target,
        'error': error,
        'prediction': prediction,
    }
    return model


def train_and_test(model, batch_size, epochs, log_dir,
                   train_input, train_output, test_input, test_output):
    """Train on the sample data."""
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(init_op)

    batch_count = int(len(train_input) / batch_size)
    print("Training set:", len(train_input))
    print("Testing set:", len(test_input))
    for i in range(epochs):
        try:
            ptr = 0
            for j in range(batch_count):
                inp, out = (
                    train_input[ptr:ptr + batch_size],
                    train_output[ptr:ptr + batch_size]
                )
                ptr += batch_size
                sess.run(
                    model['minimize'],
                    {
                        model['data']: inp, model['target']: out
                    }
                )
                # writer.add_summary(summary, i * batch_count + j)
            print("Epoch - ", str(i), datetime.now())
        except KeyboardInterrupt:
            print("Jumping out of training.")
            break
    incorrect = sess.run(
        model['error'],
        {
            model['data']: test_input, model['target']: test_output
        }
    )
    writer.flush()
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

    from IPython import embed
    embed()
    sess.close()


# From prepared CSV files
DAYS = 7
PERIODS = 48

def main(args):
    """Run all the things."""
    model = get_model((DAYS * PERIODS) + 1 , args.hidden_layers)

    n_samples, ts_in, class_out = prepare_data()
    examples = int(n_samples / 100.0 * 95)
    test_input = ts_in[examples:]
    test_output = class_out[examples:]
    train_input = ts_in[:examples]
    train_output = class_out[:examples]

    train_and_test(
        model,
        args.batch_size,
        args.epochs,
        args.log_files_path,
        train_input, train_output, test_input, test_output
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--epochs', '-e', action='store', type=int, default=1000
    )
    parser.add_argument(
        '--batch-size', '-b', action='store', type=int, default=10
    )
    parser.add_argument(
        '--hidden-layers', '-l', action='store', type=int, default=48
    )
    parser.add_argument(
        '--log-files-path', '-f', action='store', default='.'
    )
    args = parser.parse_args()
    main(args)
