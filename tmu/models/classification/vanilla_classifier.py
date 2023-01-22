# Copyright (c) 2023 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from tmu.clause_bank import ClauseBank
from tmu.models.base import TMBasis
from tmu.weight_bank import WeightBank
import numpy as np
import tmu.tools


class TMClassifier(TMBasis):
    def __init__(self, number_of_clauses, T, s, confidence_driven_updating=False, type_i_ii_ratio=1.0,
                 type_iii_feedback=False, d=200.0, platform='CPU', patch_dim=None, feature_negation=True,
                 boost_true_positive_feedback=1, max_included_literals=None, number_of_state_bits_ta=8,
                 number_of_state_bits_ind=8, weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0,
                 batch_size=100, incremental=True):
        super().__init__(number_of_clauses, T, s, confidence_driven_updating=confidence_driven_updating,
                         type_i_ii_ratio=type_i_ii_ratio, type_iii_feedback=type_iii_feedback, d=d, platform=platform,
                         patch_dim=patch_dim, feature_negation=feature_negation,
                         boost_true_positive_feedback=boost_true_positive_feedback,
                         max_included_literals=max_included_literals, number_of_state_bits_ta=number_of_state_bits_ta,
                         number_of_state_bits_ind=number_of_state_bits_ind, weighted_clauses=weighted_clauses,
                         clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p, batch_size=batch_size,
                         incremental=incremental)

    def initialize(self, X, Y):
        self.number_of_classes = int(np.max(Y) + 1)

        self.weight_banks = []
        for i in range(self.number_of_classes):
            self.weight_banks.append(WeightBank(np.concatenate((np.ones(self.number_of_clauses // 2, dtype=np.int32),
                                                                -1 * np.ones(self.number_of_clauses // 2,
                                                                             dtype=np.int32)))))

        self.clause_banks = []
        if self.platform == 'CPU':
            for i in range(self.number_of_classes):
                self.clause_banks.append(
                    ClauseBank(X, self.number_of_clauses, self.number_of_state_bits_ta, self.number_of_state_bits_ind,
                               self.patch_dim, batch_size=self.batch_size, incremental=self.incremental))
        elif self.platform == 'CUDA':
            from tmu.clause_bank_cuda import ClauseBankCUDA
            for i in range(self.number_of_classes):
                self.clause_banks.append(
                    ClauseBankCUDA(X, self.number_of_clauses, self.number_of_state_bits_ta, self.patch_dim))
        else:
            raise RuntimeError(f"Unknown platform of type: {self.platform}")

        if self.max_included_literals == None:
            self.max_included_literals = self.clause_banks[0].number_of_literals

        self.positive_clauses = np.concatenate((np.ones(self.number_of_clauses // 2, dtype=np.int32),
                                                np.zeros(self.number_of_clauses // 2, dtype=np.int32)))
        self.negative_clauses = np.concatenate((np.zeros(self.number_of_clauses // 2, dtype=np.int32),
                                                np.ones(self.number_of_clauses // 2, dtype=np.int32)))

    def fit(self, X, Y, shuffle=True):
        if self.initialized == False:
            self.initialize(X, Y)
            self.initialized = True

        if not np.array_equal(self.X_train, X):
            self.encoded_X_train = self.clause_banks[0].prepare_X(X)
            self.X_train = X.copy()

        Ym = np.ascontiguousarray(Y).astype(np.uint32)

        clause_active = []
        for i in range(self.number_of_classes):
            # Clauses are dropped based on their weights
            class_clause_active = np.ascontiguousarray(np.ones(self.number_of_clauses, dtype=np.int32))
            clause_score = np.abs(self.weight_banks[i].get_weights())
            deactivate = np.random.choice(np.arange(self.number_of_clauses),
                                          size=int(self.number_of_clauses * self.clause_drop_p),
                                          p=clause_score / clause_score.sum())
            for d in range(deactivate.shape[0]):
                class_clause_active[deactivate[d]] = 0
            clause_active.append(class_clause_active)

        # Literals are dropped based on their frequency
        literal_active = (np.zeros(self.clause_banks[0].number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
        literal_clause_frequency = self.literal_clause_frequency()
        if literal_clause_frequency.sum() > 0:
            deactivate = np.random.choice(np.arange(self.clause_banks[0].number_of_literals),
                                          size=int(self.clause_banks[0].number_of_literals * self.literal_drop_p),
                                          p=literal_clause_frequency / literal_clause_frequency.sum())
        else:
            deactivate = np.random.choice(np.arange(self.clause_banks[0].number_of_literals),
                                          size=int(self.clause_banks[0].number_of_literals * self.literal_drop_p))
        for d in range(deactivate.shape[0]):
            ta_chunk = deactivate[d] // 32
            chunk_pos = deactivate[d] % 32
            literal_active[ta_chunk] &= (~(1 << chunk_pos))

        if not self.feature_negation:
            for k in range(self.clause_banks[0].number_of_literals // 2, self.clause_banks[0].number_of_literals):
                ta_chunk = k // 32
                chunk_pos = k % 32
                literal_active[ta_chunk] &= (~(1 << chunk_pos))
        literal_active = literal_active.astype(np.uint32)

        shuffled_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(shuffled_index)

        for e in shuffled_index:
            target = Ym[e]

            clause_outputs = self.clause_banks[target].calculate_clause_outputs_update(literal_active,
                                                                                       self.encoded_X_train, e)
            class_sum = np.dot(clause_active[target] * self.weight_banks[target].get_weights(), clause_outputs).astype(
                np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)

            if self.confidence_driven_updating:
                update_p = 1.0 * (self.T - np.absolute(class_sum)) / self.T
            else:
                update_p = (self.T - class_sum) / (2 * self.T)

            if self.weighted_clauses:
                self.weight_banks[target].increment(clause_outputs, update_p, clause_active[target], False)
            self.clause_banks[target].type_i_feedback(update_p * self.type_i_p, self.s,
                                                      self.boost_true_positive_feedback, self.max_included_literals,
                                                      clause_active[target] * self.positive_clauses, literal_active,
                                                      self.encoded_X_train, e)
            self.clause_banks[target].type_ii_feedback(update_p * self.type_ii_p,
                                                       clause_active[target] * self.negative_clauses, literal_active,
                                                       self.encoded_X_train, e)
            if self.type_iii_feedback:
                self.clause_banks[target].type_iii_feedback(update_p, self.d,
                                                            clause_active[target] * self.positive_clauses,
                                                            literal_active, self.encoded_X_train, e, 1)
                self.clause_banks[target].type_iii_feedback(update_p, self.d,
                                                            clause_active[target] * self.negative_clauses,
                                                            literal_active, self.encoded_X_train, e, 0)

            not_target = np.random.randint(self.number_of_classes)
            while not_target == target:
                not_target = np.random.randint(self.number_of_classes)

            clause_outputs = self.clause_banks[not_target].calculate_clause_outputs_update(literal_active,
                                                                                           self.encoded_X_train, e)
            class_sum = np.dot(clause_active[not_target] * self.weight_banks[not_target].get_weights(),
                               clause_outputs).astype(np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)

            if self.confidence_driven_updating:
                update_p = 1.0 * (self.T - np.absolute(class_sum)) / self.T
            else:
                update_p = (self.T + class_sum) / (2 * self.T)

            if self.weighted_clauses:
                self.weight_banks[not_target].decrement(clause_outputs, update_p, clause_active[not_target], False)
            self.clause_banks[not_target].type_i_feedback(update_p * self.type_i_p, self.s,
                                                          self.boost_true_positive_feedback, self.max_included_literals,
                                                          clause_active[not_target] * self.negative_clauses,
                                                          literal_active, self.encoded_X_train, e)
            self.clause_banks[not_target].type_ii_feedback(update_p * self.type_ii_p,
                                                           clause_active[not_target] * self.positive_clauses,
                                                           literal_active, self.encoded_X_train, e)
            if self.type_iii_feedback:
                self.clause_banks[not_target].type_iii_feedback(update_p, self.d,
                                                                clause_active[not_target] * self.negative_clauses,
                                                                literal_active, self.encoded_X_train, e, 1)
                self.clause_banks[not_target].type_iii_feedback(update_p, self.d,
                                                                clause_active[not_target] * self.positive_clauses,
                                                                literal_active, self.encoded_X_train, e, 0)
        return

    def predict(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_banks[0].prepare_X(X)
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))
        for e in range(X.shape[0]):
            max_class_sum = -self.T
            max_class = 0
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.weight_banks[i].get_weights(),
                                   self.clause_banks[i].calculate_clause_outputs_predict(self.encoded_X_test,
                                                                                         e)).astype(np.int32)
                class_sum = np.clip(class_sum, -self.T, self.T)
                if class_sum > max_class_sum:
                    max_class_sum = class_sum
                    max_class = i
            Y[e] = max_class
        return Y

    def transform(self, X):
        encoded_X = self.clause_banks[0].prepare_X(X)
        transformed_X = np.empty((X.shape[0], self.number_of_classes, self.number_of_clauses), dtype=np.uint32)
        for e in range(X.shape[0]):
            for i in range(self.number_of_classes):
                transformed_X[e, i, :] = self.clause_banks[i].calculate_clause_outputs_predict(encoded_X, e)
        return transformed_X.reshape((X.shape[0], self.number_of_classes * self.number_of_clauses))

    def transform_patchwise(self, X):
        encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                     self.patch_dim, 0)
        transformed_X = np.empty(
            (X.shape[0], self.number_of_classes, self.number_of_clauses // 2 * self.number_of_patches), dtype=np.uint32)
        for e in range(X.shape[0]):
            for i in range(self.number_of_classes):
                transformed_X[e, i, :] = self.clause_bank[i].calculate_clause_outputs_patchwise(encoded_X, e)
        return transformed_X.reshape(
            (X.shape[0], self.number_of_classes * self.number_of_clauses, self.number_of_patches))

    def literal_clause_frequency(self):
        clause_active = np.ones(self.number_of_clauses, dtype=np.uint32)
        literal_frequency = np.zeros(self.clause_banks[0].number_of_literals, dtype=np.uint32)
        for i in range(self.number_of_classes):
            literal_frequency += self.clause_banks[i].calculate_literal_clause_frequency(clause_active)
        return literal_frequency

    def literal_importance(self, the_class, negated_features=False, negative_polarity=False):
        literal_frequency = np.zeros(self.clause_banks[0].number_of_literals, dtype=np.uint32)
        if negated_features:
            if negative_polarity:
                literal_frequency[self.clause_banks[the_class].number_of_literals // 2:] += self.clause_banks[
                                                                                                the_class].calculate_literal_clause_frequency(
                    self.negative_clauses)[self.clause_banks[the_class].number_of_literals // 2:]
            else:
                literal_frequency[self.clause_banks[the_class].number_of_literals // 2:] += self.clause_banks[
                                                                                                the_class].calculate_literal_clause_frequency(
                    self.positive_clauses)[self.clause_banks[the_class].number_of_literals // 2:]
        else:
            if negative_polarity:
                literal_frequency[:self.clause_banks[the_class].number_of_literals // 2] += self.clause_banks[
                                                                                                the_class].calculate_literal_clause_frequency(
                    self.negative_clauses)[:self.clause_banks[the_class].number_of_literals // 2]
            else:
                literal_frequency[:self.clause_banks[the_class].number_of_literals // 2] += self.clause_banks[
                                                                                                the_class].calculate_literal_clause_frequency(
                    self.positive_clauses)[:self.clause_banks[the_class].number_of_literals // 2]

        return literal_frequency

    def clause_precision(self, the_class, polarity, X, Y):
        clause_outputs = self.transform(X).reshape(X.shape[0], self.number_of_classes, 2, self.number_of_clauses // 2)[
                         :, the_class, polarity, :]
        if polarity == 0:
            true_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
        else:
            true_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)
        return np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0,
                        true_positive_clause_outputs / (true_positive_clause_outputs + false_positive_clause_outputs))

    def clause_recall(self, the_class, polarity, X, Y):
        clause_outputs = self.transform(X).reshape(X.shape[0], self.number_of_classes, 2, self.number_of_clauses // 2)[
                         :, the_class, polarity, :]
        if polarity == 0:
            true_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0) / Y[Y == the_class].shape[0]
        else:
            true_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0) / Y[Y != the_class].shape[0]
        return true_positive_clause_outputs

    def get_weight(self, the_class, polarity, clause):
        if polarity == 0:
            return self.weight_banks[the_class].get_weights()[clause]
        else:
            return self.weight_banks[the_class].get_weights()[self.number_of_clauses // 2 + clause]

    def set_weight(self, the_class, polarity, clause, weight):
        if polarity == 0:
            self.weight_banks[the_class].get_weights()[clause] = weight
        else:
            self.weight_banks[the_class].get_weights()[self.number_of_clauses // 2 + clause] = weight

    def get_ta_action(self, the_class, polarity, clause, ta):
        if polarity == 0:
            return self.clause_banks[the_class].get_ta_action(clause, ta)
        else:
            return self.clause_banks[the_class].get_ta_action(self.number_of_clauses // 2 + clause, ta)

    def get_ta_state(self, the_class, polarity, clause, ta):
        if polarity == 0:
            return self.clause_banks[the_class].get_ta_state(clause, ta)
        else:
            return self.clause_banks[the_class].get_ta_state(self.number_of_clauses // 2 + clause, ta)

    def set_ta_state(self, the_class, polarity, clause, ta, state):
        if polarity == 0:
            return self.clause_banks[the_class].set_ta_state(clause, ta, state)
        else:
            return self.clause_banks[the_class].set_ta_state(self.number_of_clauses // 2 + clause, ta, state)

    def number_of_include_actions(self, the_class, clause):
        return self.clause_banks[the_class].number_of_include_actions(clause)