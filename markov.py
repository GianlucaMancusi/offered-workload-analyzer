"""
    Lavoro svolto per un progetto di Sistemi ed Applicazioni Cloud
    Gianluca Mancusi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MarkovAnalysis():
    def __init__(self, states, initial_state, offered_weights):
        super().__init__()
        self.states = states
        self.matrix = pd.DataFrame(data=None, index=states, columns=states)
        self.initial_state = initial_state
        self.states_probabilities = pd.Series(index=states)
        self.states_probabilities[self.initial_state] = 1
        self.transition_history = pd.DataFrame()
        self.offered_weights = offered_weights

    def add_transition(self, start_state, end_state, probability):
        if start_state not in self.states or end_state not in self.states:
            raise AssertionError("start_state and end_state must be valid")

        self.matrix.at[end_state, start_state] = probability

    def show_history(self):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.transition_history)

    def show_matrix(self):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.matrix)

    def run_analysis(self, num_iterations):
        if num_iterations <= 0:
            raise AssertionError("The number of iterations must be > 0")

        self.matrix.fillna(value=0, inplace=True)
        self.states_probabilities.fillna(value=0, inplace=True)
        self.states_probabilities_weighted = pd.Series(self.states_probabilities)
        result = pd.Series(self.states_probabilities)
        result_weighted = pd.Series(self.states_probabilities)

        self.transition_history = pd.DataFrame()

        for it in range(num_iterations):

            # Save the history
            self.transition_history[f"step {it+1}"] = result

            # Compute the result
            result = self.matrix.dot(result)
            result_weighted = self.matrix.dot(result) * self.offered_weights

            # Normalize the result
            result = result / result.sum()
            result_weighted = result_weighted / result_weighted.sum()

        self.states_probabilities = result
        self.states_probabilities_weighted = result_weighted
        return self.states_probabilities


if __name__ == "__main__":

    states = [
              "Ricerca", "Naviga", "Visualizzare\nristorante", "Exit", "Aggiungere\npiatti",
              "Paga","Scrivere\nrecensione", "Caricamento file\nmultimediali","Entry", "Social Login", "SMS Login", ]
    initial_state = "Entry"
    iterations = 11

    offered_weight = [1, 1, 1, 1, 1, 1, 1, 150, 1, 1, 1]

    markov = MarkovAnalysis(states, initial_state, offered_weight)

    # Adding transition probabilites
    markov.add_transition("Entry",                          "Ricerca",                          0.40)
    markov.add_transition("Entry",                          "Naviga",                           0.40)
    markov.add_transition("Entry",                          "Social Login",                     0.15)
    markov.add_transition("Entry",                          "SMS Login",                        0.05)
    markov.add_transition("Social Login",                   "Ricerca",                          0.45)
    markov.add_transition("Social Login",                   "Naviga",                           0.45)
    markov.add_transition("SMS Login",                      "Ricerca",                          0.45)
    markov.add_transition("SMS Login",                      "Naviga",                           0.45)
    markov.add_transition("Naviga",                         "Naviga",                           0.15)
    markov.add_transition("Naviga",                         "Ricerca",                          0.60)
    markov.add_transition("Naviga",                         "Visualizzare\nristorante",         0.15)
    markov.add_transition("Ricerca",                        "Ricerca",                          0.40)
    markov.add_transition("Ricerca",                        "Visualizzare\nristorante",         0.30)
    markov.add_transition("Ricerca",                        "Naviga",                           0.20)
    markov.add_transition("Visualizzare\nristorante",       "Ricerca",                          0.10)
    markov.add_transition("Visualizzare\nristorante",       "Naviga",                           0.40)
    markov.add_transition("Visualizzare\nristorante",       "Aggiungere\npiatti",               0.30)
    markov.add_transition("Visualizzare\nristorante",       "Scrivere\nrecensione",             0.10)
    markov.add_transition("Scrivere\nrecensione",           "Visualizzare\nristorante",         0.30)
    markov.add_transition("Scrivere\nrecensione",           "Caricamento file\nmultimediali",   0.50)
    markov.add_transition("Aggiungere\npiatti",             "Visualizzare\nristorante",         0.45)
    markov.add_transition("Aggiungere\npiatti",             "Paga",                             0.45)
    markov.add_transition("Paga",                           "Naviga",                           0.25)
    markov.add_transition("Paga",                           "Ricerca",                          0.25)

    # Exit probabilites
    markov.add_transition("Social Login",                   "Exit",                             0.10)
    markov.add_transition("SMS Login",                      "Exit",                             0.10)
    markov.add_transition("Ricerca",                        "Exit",                             0.10)
    markov.add_transition("Naviga",                         "Exit",                             0.10)
    markov.add_transition("Visualizzare\nristorante",       "Exit",                             0.10)
    markov.add_transition("Scrivere\nrecensione",           "Exit",                             0.20)
    markov.add_transition("Caricamento file\nmultimediali", "Exit",                             0.40)
    markov.add_transition("Aggiungere\npiatti",             "Exit",                             0.10)
    markov.add_transition("Paga",                           "Exit",                             0.50)

    # Show the DataFrame

    markov.show_matrix()

    # Run the markov analysis
    final_result = markov.run_analysis(num_iterations=iterations)

    print(final_result)

    markov.show_history()

    ax = pd.DataFrame(markov.states_probabilities_weighted[[0,1,2,7]]).plot(kind='bar', rot=45)
    ax.set_xlabel("Azioni dell'utente")
    ax.legend(["Peso sul network moltiplicato per la frequenza"])
    plt.show()

    ax = pd.DataFrame(final_result).plot(kind='bar', rot=45)
    ax.set_xlabel("Azioni dell'utente")
    ax.legend(["Probabilità percentuale"])
    plt.show()
