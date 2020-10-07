from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import shap
import matplotlib.pyplot as plt


def dense_2_sigmoid_binary_crossent():
    model = Sequential([
        Dense(32, input_shape=(2,), activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def dense_2_softmax_categ_crossent():
    model = Sequential([
        Dense(32, input_shape=(2,), activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def dataset(n):
    # x is a price tag, y is 1 if expensive (>=100), 0 if cheap (<100)
    x = np.random.normal(loc=100, scale=50, size=(n, 2)).astype(int)
    y0 = np.reshape((x[:, 0] >= 100).astype(int), (n, 1))
    y1 = np.reshape((x[:, 0] < 100).astype(int), (n, 1))
    y = np.concatenate((y0, y1), axis=1)
    return x, y


def plot_dot(title, x=None):
    plt.figure()

    if x is None:
        x = x_test
    shap.summary_plot(shap_values[0], features=x,
                      feature_names=feature_names,
                      class_names=class_names[0], title=class_names[0], show=False)

    plt.gca().set_title(title)
    plt.tight_layout()
    plt.savefig("{}.png".format(title), bbox_inches='tight')


if __name__ == "__main__":
    feature_names = ['x0', 'x1']
    class_names = ['y0', 'y1']

    x_train, y_train = dataset(1000)
    x_test, y_test = dataset(1000)

    plt.scatter(x_train[:, 0], y_train[:, 0])
    plt.gca().set_title("dataset")
    plt.tight_layout()
    plt.savefig("dataset.png", bbox_inches='tight')

    model = dense_2_sigmoid_binary_crossent()
    model.fit(x_train, y_train, epochs=20)

    e = shap.DeepExplainer(model, x_train)
    shap_values = e.shap_values(x_test)
    plot_dot(title="DeepExplainer, sigmoid, y0")

    e = shap.GradientExplainer(model, x_train)
    shap_values = e.shap_values(x_test, nsamples=x_train.shape[0])
    plot_dot(title="GradientExplainer, sigmoid, y0")

    e = shap.KernelExplainer(model.predict, x_train[:100, ])
    shap_values = e.shap_values(x_test[:100, ], nsamples=100)
    plot_dot(title="KernelExplainer, sigmoid, y0", x=x_test[:100, ])

    model = dense_2_softmax_categ_crossent()
    model.fit(x_train, y_train, epochs=20)

    e = shap.DeepExplainer(model, x_train)
    shap_values = e.shap_values(x_test)
    plot_dot(title="DeepExplainer, softmax, y0")

    e = shap.GradientExplainer(model, x_train)
    shap_values = e.shap_values(x_test, nsamples=x_train.shape[0])
    plot_dot(title="GradientExplainer, softmax, y0")

    e = shap.KernelExplainer(model.predict, x_train[:100, ])
    shap_values = e.shap_values(x_test[:100, ], nsamples=x_train.shape[0])
    plot_dot(title="KernelExplainer, softmax, y0", x=x_test[:100, ])
