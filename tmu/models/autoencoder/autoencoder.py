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
import random
from tabulate import tabulate
from tmu.weight_bank import WeightBank
from tmu.models.base import MultiWeightBankMixin, SingleClauseBankMixin, TMBaseModel
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

class TMAutoEncoder(TMBaseModel, SingleClauseBankMixin, MultiWeightBankMixin):
    def __init__(
            self,
            number_of_clauses,
            T,
            s,
            output_active,
            accumulation=1,
            type_i_ii_ratio=1.0,
            type_iii_feedback=False,
            focused_negative_sampling=False,
            output_balancing=0,
            upsampling=1,
            d=200.0,
            platform='CPU',
            patch_dim=None,
            feature_negation=True,
            boost_true_positive_feedback=1,
            reuse_random_feedback=0,
            max_included_literals=None,
            number_of_state_bits_ta=8,
            number_of_state_bits_ind=8,
            weighted_clauses=False,
            clause_drop_p=0.0,
            literal_drop_p=0.0,
            absorbing=-1,
            literal_sampling=1.0,
            feedback_rate_excluded_literals=1,
            literal_insertion_state=-1,
            squared_weight_update_p=False,
            seed=None
    ):
        self.output_active = output_active
        self.accumulation = accumulation
        super().__init__(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            type_i_ii_ratio=type_i_ii_ratio,
            type_iii_feedback=type_iii_feedback,
            focused_negative_sampling=focused_negative_sampling,
            output_balancing=output_balancing,
            upsampling=upsampling,
            d=d,
            platform=platform, patch_dim=patch_dim,
            feature_negation=feature_negation,
            boost_true_positive_feedback=boost_true_positive_feedback,
            reuse_random_feedback=reuse_random_feedback,
            max_included_literals=max_included_literals,
            number_of_state_bits_ta=number_of_state_bits_ta,
            number_of_state_bits_ind=number_of_state_bits_ind,
            weighted_clauses=weighted_clauses,
            clause_drop_p=clause_drop_p,
            literal_drop_p=literal_drop_p,
            absorbing=absorbing,
            literal_sampling=literal_sampling,
            feedback_rate_excluded_literals=feedback_rate_excluded_literals,
            literal_insertion_state=literal_insertion_state,
            squared_weight_update_p=squared_weight_update_p,
            seed=seed
        )
        SingleClauseBankMixin.__init__(self)
        MultiWeightBankMixin.__init__(self, seed=seed)
        self.max_positive_clauses = number_of_clauses

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        clause_bank_type, clause_bank_args = self.build_clause_bank(X=X)
        self.clause_bank = clause_bank_type(**clause_bank_args)

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        self.number_of_classes = self.output_active.shape[0]
        self.weight_banks.set_clause_init(WeightBank, dict(
            weights=self.rng.choice([-1, 1], size=self.number_of_clauses).astype(np.int32)
        ))
        self.weight_banks.populate(list(range(self.number_of_classes)))

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        if self.max_included_literals is None:
            self.max_included_literals = self.clause_bank.number_of_literals

        if self.max_positive_clauses is None:
            self.max_positive_clauses = self.number_of_clauses

        if self.output_balancing == 0:
            self.feature_true_probability = np.asarray(X.sum(axis=0)/X.shape[0]).reshape(-1)**(1.0/self.upsampling)
        else:
            self.feature_true_probability = np.ones(X.shape[1], dtype=np.float32)*self.output_balancing

    def update(
            self,
            target_output,
            Y,
            encoded_X,
            clause_active,
            literal_active,
            weights = None
    ):
        all_literal_active = (np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32) | ~0).astype(np.uint32)
        clause_outputs = self.clause_bank.calculate_clause_outputs_update(all_literal_active, encoded_X, 0)

        target_output_weights = self.weight_banks[target_output].get_weights()
        
        if weights != None:
            for i, target_weight in enumerate(target_output_weights):
                target_output_weights[i] = weights[i]
            # self.weight_banks[target_output].set_weight(Y_weight)

        class_sum = np.dot(clause_active * target_output_weights, clause_outputs).astype(
            np.int32)
        class_sum = np.clip(class_sum, -self.T, self.T)

        type_iii_feedback_selection = self.rng.choice(2)

        if Y == 1:
            update_p = (self.T - class_sum) / (2 * self.T)
            if self.squared_weight_update_p:
                update_p = update_p ** 2

            self.clause_bank.type_i_feedback(
                update_p=update_p * self.type_i_p,
                clause_active=clause_active * (target_output_weights >= 0),
                literal_active=literal_active,
                encoded_X=encoded_X,
                e=0
            )

            self.clause_bank.type_ii_feedback(
                update_p=update_p * self.type_ii_p,
                clause_active=clause_active * (target_output_weights < 0),
                literal_active=literal_active,
                encoded_X=encoded_X,
                e=0
            )

            self.weight_banks[target_output].increment(
                clause_output=clause_outputs,
                update_p=update_p,
                clause_active=clause_active,
                positive_weights=True
            )

            if self.type_iii_feedback and type_iii_feedback_selection == 0:
                self.clause_bank.type_iii_feedback(
                    update_p=update_p,
                    clause_active=clause_active * (target_output_weights >= 0),
                    literal_active=literal_active,
                    encoded_X=encoded_X,
                    e=0,
                    target=1
                )

                self.clause_bank.type_iii_feedback(
                    update_p=update_p,
                    clause_active=clause_active * (target_output_weights < 0),
                    literal_active=literal_active,
                    encoded_X=encoded_X,
                    e=0,
                    target=0
                )
        else:
            update_p = (self.T + class_sum) / (2 * self.T)
            if self.squared_weight_update_p:
                update_p = update_p ** 2

            self.clause_bank.type_i_feedback(
                update_p=update_p * self.type_i_p,
                clause_active=clause_active * (target_output_weights < 0),
                literal_active=literal_active,
                encoded_X=encoded_X,
                e=0
            )

            self.clause_bank.type_ii_feedback(
                update_p=update_p * self.type_ii_p,
                clause_active=clause_active * (target_output_weights >= 0),
                literal_active=literal_active,
                encoded_X=encoded_X,
                e=0
            )

            self.weight_banks[target_output].decrement(
                clause_output=clause_outputs,
                update_p=update_p,
                clause_active=clause_active,
                negative_weights=True
            )

            if self.type_iii_feedback and type_iii_feedback_selection == 1:
                self.clause_bank.type_iii_feedback(
                    update_p=update_p,
                    clause_active=clause_active * (target_output_weights < 0),
                    literal_active=literal_active,
                    encoded_X=encoded_X,
                    e=0,
                    target=1
                )

                self.clause_bank.type_iii_feedback(
                    update_p=update_p,
                    clause_active=clause_active * (target_output_weights >= 0),
                    literal_active=literal_active,
                    encoded_X=encoded_X,
                    e=0,
                    target=0
                )

    def activate_clauses(self):
        # Drops clauses randomly based on clause drop probability
        clause_active = (self.rng.rand(self.number_of_clauses) >= self.clause_drop_p).astype(np.int32)

        return clause_active

    def activate_literals(self):
        # Literals are dropped based on literal drop probability
        literal_active = np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32)
        literal_active_integer = self.rng.rand(self.clause_bank.number_of_literals) >= self.literal_drop_p
        for k in range(self.clause_bank.number_of_literals):
            if literal_active_integer[k] == 1:
                ta_chunk = k // 32
                chunk_pos = k % 32
                literal_active[ta_chunk] |= (1 << chunk_pos)

        if not self.feature_negation:
            for k in range(self.clause_bank.number_of_literals // 2, self.clause_bank.number_of_literals):
                ta_chunk = k // 32
                chunk_pos = k % 32
                literal_active[ta_chunk] &= (~(1 << chunk_pos))
        literal_active = literal_active.astype(np.uint32)

        return literal_active

    def documents_fit(
            self, 
            X, 
            number_of_examples=2000, 
            categories=0,
            random_per_category=False,
            involved_datasets=[],
            shuffle=True, 
            proportional_ds = False,
            print_python = False,
            print_c = False,
            *kwargs
            ):
        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        X_csc = csc_matrix(X.reshape(X.shape[0], -1)).sorted_indices()
        self.init(X_csr, Y=None)

        if not np.array_equal(self.X_train, np.concatenate((X_csr.indptr, X_csr.indices))):
            self.encoded_X_train = self.clause_bank.prepare_X_autoencoder(X_csr, X_csc, self.output_active)
            self.X_train = np.concatenate((X_csr.indptr, X_csr.indices))

        clause_active = self.activate_clauses()
        literal_active = self.activate_literals()

        class_index = np.arange(self.number_of_classes, dtype=np.uint32)

        total_size = sum(dataset[2] - dataset[1] for dataset in involved_datasets) 
        number_of_experts = len(involved_datasets)
        examples_per_expert = [] 
        for dataset in involved_datasets:
            if proportional_ds:
                size = dataset[2] - dataset[1]
                examples = int(number_of_examples * (size / total_size))
            else:
                examples = int(number_of_examples / number_of_experts)
            examples_per_expert.append(examples)

        if number_of_experts > 1:
            self.custom_print(print_python,"Number of Experts = %s" % number_of_experts)
        expert_start_index = 0
        expert_end_index=0
        ex = 0
        for expert in examples_per_expert:
            self.custom_print(print_python,"Running experts: %d, with number of examples:%d" % (ex+1,expert))
            if(number_of_experts > 1):
                expert_start_index = involved_datasets[ex][1]
                expert_end_index = involved_datasets[ex][2]
                ex = ex + 1

            for _ in range(expert):
                self.rng.shuffle(class_index)

                average_absolute_weights = np.zeros(self.number_of_clauses, dtype=np.float32)
                for i in class_index:
                    average_absolute_weights += np.absolute(self.weight_banks[i].get_weights())
                average_absolute_weights /= self.number_of_classes
                update_clause = self.rng.random(self.number_of_clauses) <= (
                        self.T - np.clip(average_absolute_weights, 0, self.T)) / self.T

                Xu, Yu = self.clause_bank.produce_autoencoder_example(X_csr, 
                                                                    X_csc, 
                                                                    self.output_active, 
                                                                    self.accumulation, 
                                                                    categories = categories, 
                                                                    random_per_category = random_per_category,
                                                                    expert_start_index=expert_start_index,
                                                                    expert_end_index=expert_end_index,
                                                                    enable_c_log=print_c)         
                for i in class_index:
                    (target, encoded_X) = Yu[i], Xu[i].reshape((1, -1))

                    ta_chunk = self.output_active[i] // 32
                    chunk_pos = self.output_active[i] % 32
                    copy_literal_active_ta_chunk = literal_active[ta_chunk]

                    if self.feature_negation:
                        ta_chunk_negated = (self.output_active[i] + self.clause_bank.number_of_features) // 32
                        chunk_pos_negated = (self.output_active[i] + self.clause_bank.number_of_features) % 32
                        copy_literal_active_ta_chunk_negated = literal_active[ta_chunk_negated]
                        literal_active[ta_chunk_negated] &= ~(1 << chunk_pos_negated)

                    literal_active[ta_chunk] &= ~(1 << chunk_pos)

                    self.update(i, target, encoded_X, update_clause * clause_active, literal_active)        

                    if self.feature_negation:
                        literal_active[ta_chunk_negated] = copy_literal_active_ta_chunk_negated
                    literal_active[ta_chunk] = copy_literal_active_ta_chunk
        return
 
    def clauses_fit(
            self, 
            number_of_examples,
            number_of_features,
            target_words_clauses,
            negative_weight_clause,
            cross_accumlation = False,
            weight_insertion = False,
            print_python = False,
            print_c = False
            ):

        #target_words_clauses=
        #[0,[[clause1],[clause2],[clause3],...,[clauseN]]]
        #[1,[[clause1],[clause2],[clause3],...,[clauseN]]]
        #[2,[[clause1],[clause2],[clause3],...,[clauseN]]]
        #[...,[[clause1],[clause2],[clause3],...,[clauseN]]]
        #[TWs,[[clause1],[clause2],[clause3],...,[clauseN]]]

        #clause = weight,[feature1, feature2, ...]

        max_feature = self.calc_max_no_features(number_of_features, target_words_clauses, print_python)
        X_csc = csr_matrix((1, max_feature), dtype=np.int64)
        self.init(X=X_csc, Y=None)
        clause_active = self.activate_clauses()
        literal_active = self.activate_literals()
        class_index = np.arange(self.number_of_classes, dtype=np.uint32)

        for ex in range(number_of_examples):
            self.rng.shuffle(class_index)

            average_absolute_weights = np.zeros(self.number_of_clauses, dtype=np.float32)
            for i in class_index:
                average_absolute_weights += np.absolute(self.weight_banks[i].get_weights())
            average_absolute_weights /= self.number_of_classes
            update_clause = self.rng.random(self.number_of_clauses) <= (
                    self.T - np.clip(average_absolute_weights, 0, self.T)) / self.T

            for i in class_index:
                source_clauses, source_clauses_weights, source_max_columns = self.prepare_clauses(target_words_clauses, print_python, target_word = i)

                weights = None
                if cross_accumlation:
                    for j in class_index:
                        if i != j:
                            destination_clauses, destination_clauses_weights, destination_max_columns = self.prepare_clauses(target_words_clauses, print_python, target_word = j)
                            Xu, Yu = self.clause_bank.produce_autoencoder_combined(
                                target_true_p=self.feature_true_probability[self.output_active[i]],
                                accumulation=self.accumulation,
                                no_of_involved_fetures = max_feature,
                                source_clauses = source_clauses,
                                source_clauses_weights = source_clauses_weights,
                                source_no_columns = int(source_max_columns),
                                destination_clauses = destination_clauses,
                                destination_clauses_weights = destination_clauses_weights,
                                destination_no_columns = int(destination_max_columns),
                                negative_weight_clause = negative_weight_clause,
                                enable_c_log = print_c
                            )        
                            self.update_from_clauses(update_clause * clause_active, literal_active, i, update_clause, Xu, Yu, weights)      
                else:
                    if weight_insertion:
                        weights = []
                        number_of_ta_chunks = int(((max_feature * 2) - 1) / 32 + 1)
                        X = np.ascontiguousarray(np.empty(number_of_ta_chunks, dtype=np.uint32))
                        Yu = random.randint(0, 1)
                        weights_indeces = []
                        if Yu == 1:
                            for index, weight in enumerate(source_clauses_weights):
                                if weight > 0:
                                    weights_indeces.append(index)
                        else:
                            for index, weight in enumerate(source_clauses_weights):
                                if weight <= 0:
                                    weights_indeces.append(index)
                        for j in range(self.number_of_clauses):
                            random_index = random.choice(weights_indeces)
                            weights.append(source_clauses_weights[random_index])
                            self.store_to_X(number_of_features, source_clauses[random_index], X)
                        Xu = X.reshape((1, -1))
                        if(ex > 0):
                            weights = None
                        self.update_from_clauses(update_clause * clause_active, literal_active, i, update_clause, Xu, Yu, weights)      
                    else:
                        Xu, Yu = self.clause_bank.produce_autoencoder_from_clauses(
                            target_true_p=self.feature_true_probability[self.output_active[i]],
                            accumulation=self.accumulation,
                            number_of_features = max_feature,
                            source_clauses = source_clauses,
                            source_clauses_weights = source_clauses_weights,
                            source_no_columns = int(source_max_columns),
                            negative_weight_clause = negative_weight_clause,
                            enable_c_log = print_c
                        )
                        self.update_from_clauses(update_clause * clause_active, literal_active, i, update_clause, Xu, Yu, weights)      

        return

    def clauses_pairs_fit(
            self, 
            number_of_examples,
            number_of_features,
            target_words_clauses,
            target_words_pairs,
            negative_weight_clause = True,
            print_python = False,
            print_c = False
            ):
        
        max_feature = self.calc_max_no_features(number_of_features, target_words_clauses, print_python)
        X_csc = csr_matrix((1, max_feature), dtype=np.int64)
        self.init(X=X_csc, Y=None)
        clause_active = self.activate_clauses()
        
        for target in target_words_clauses:
            target_word_clauses = target[1]
            for clause in target_word_clauses:
                related_literals = clause[1]
                for feature in related_literals:
                    if feature >= number_of_features:
                        feature = -1 * (feature - number_of_features)
                    ta_chunk = feature // 32
                    chunk_pos = feature % 32
                    self.literal_active[ta_chunk] |= (1 << chunk_pos)

        class_index = np.arange(self.number_of_classes, dtype=np.uint32)

        for ex in range(number_of_examples):
            self.rng.shuffle(class_index)
            self.rng.shuffle(target_words_pairs)

            average_absolute_weights = np.zeros(self.number_of_clauses, dtype=np.float32)
            for i in class_index:
                average_absolute_weights += np.absolute(self.weight_banks[i].get_weights())
            average_absolute_weights /= self.number_of_classes
            update_clause = self.rng.random(self.number_of_clauses) <= (
                    self.T - np.clip(average_absolute_weights, 0, self.T)) / self.T

            for source, destination in target_words_pairs:
                source_clauses, source_clauses_weights, source_max_columns = self.prepare_clauses(target_words_clauses, print_python, target_word = source)
                destination_clauses, destination_clauses_weights, destination_max_columns = self.prepare_clauses(target_words_clauses, print_python, target_word = destination)
                Xu, Yu = self.clause_bank.produce_autoencoder_combined(
                    target_true_p=self.feature_true_probability[self.output_active[i]],
                    accumulation=self.accumulation,
                    no_of_involved_fetures = max_feature,
                    source_clauses = source_clauses,
                    source_clauses_weights = source_clauses_weights,
                    source_no_columns = int(source_max_columns),
                    destination_clauses = destination_clauses,
                    destination_clauses_weights = destination_clauses_weights,
                    destination_no_columns = int(destination_max_columns),
                    negative_weight_clause = negative_weight_clause,
                    enable_c_log = print_c
                )        
                self.update_from_clauses(update_clause * clause_active, literal_active, source, Xu, Yu, weights = None)      
                self.update_from_clauses(update_clause * clause_active, literal_active, destination, Xu, Yu, weights = None)      
        return

    def calc_max_no_features(self, number_of_features, target_words_clauses, print_python):
        literals = 0
        max_feature = number_of_features
        for target in target_words_clauses:
            target_word_clauses = target[1]
            for clause in target_word_clauses:
                related_literals = clause[1]
                literals += len(related_literals)
                
                for feature in related_literals:
                    if feature >= number_of_features:
                        feature = -1 * (feature - number_of_features)
                    if feature > max_feature:
                        max_feature = feature
        self.custom_print(print_python,"Count of all related_literals:", literals)
        self.custom_print(print_python,"Maximum feature:", max_feature)
        return max_feature

    def update_from_clauses(self, clause_active, literal_active, i, Xu, Yu, weights):
        ta_chunk = self.output_active[i] // 32
        chunk_pos = self.output_active[i] % 32
        copy_literal_active_ta_chunk = literal_active[ta_chunk]

        if self.feature_negation:
            ta_chunk_negated = (self.output_active[i] + self.clause_bank.number_of_features) // 32
            chunk_pos_negated = (self.output_active[i] + self.clause_bank.number_of_features) % 32
            copy_literal_active_ta_chunk_negated = literal_active[ta_chunk_negated]
            literal_active[ta_chunk_negated] &= ~(1 << chunk_pos_negated)

        literal_active[ta_chunk] &= ~(1 << chunk_pos)

        self.update(i, Yu, Xu, clause_active, literal_active, weights = weights)

        if self.feature_negation:
            literal_active[ta_chunk_negated] = copy_literal_active_ta_chunk_negated
        literal_active[ta_chunk] = copy_literal_active_ta_chunk

    def store_to_X(self, number_of_features, clause, X):
        for feature in clause:
            chunk_nr = feature // 32
            chunk_pos = feature % 32
            X[chunk_nr] |= (1 << chunk_pos)

            chunk_nr = (feature + number_of_features) // 32
            chunk_pos = (feature + number_of_features) % 32
            X[chunk_nr] &= ~(1 << chunk_pos)

    def prepare_clauses(self, target_words_clauses, print_python, target_word):
        clauses = []
        clauses_weights = []
        for clauses_array in target_words_clauses[target_word][1]:
            clauses.append(clauses_array[1])
            clauses_weights.append(clauses_array[0])

        max_columns = max(len(row) for row in clauses)
        for clause in clauses:
            while len(clause) < max_columns:
                clause.append(0)

        if print_python:
            header = [f"Column {i}" for i in range(1, max_columns + 1)]
            table = [row for row in clauses]
            self.custom_print(print_python,tabulate(table, headers=header, tablefmt="grid"))

        return clauses,clauses_weights,max_columns

    def custom_print(*args, **kwargs):
        enabled = kwargs.pop('enabled', False)
        if enabled:
            print(*args, **kwargs)

    def predict(self, X, **kwargs):
        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        Y = np.ascontiguousarray(np.zeros((X.shape[0], self.number_of_classes), dtype=np.uint32))

        for e in range(X.shape[0]):
            encoded_X = self.clause_bank.prepare_X(X_csr[e, :].toarray())

            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(encoded_X, 0)
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
                Y[e, i] = (class_sum >= 0)
        return Y

    def literal_importance(self, the_class, negated_features=False, negative_polarity=False):
        literal_frequency = np.zeros(self.clause_bank.number_of_literals, dtype=np.uint32)
        if negated_features:
            if negative_polarity:
                literal_frequency[
                self.clause_bank.number_of_literals // 2:] += self.clause_bank.calculate_literal_clause_frequency(
                    self.weight_banks[the_class].get_weights() < 0)[self.clause_bank.number_of_literals // 2:]
            else:
                literal_frequency[
                self.clause_bank.number_of_literals // 2:] += self.clause_bank.calculate_literal_clause_frequency(
                    self.weight_banks[the_class].get_weights() >= 0)[self.clause_bank.number_of_literals // 2:]
        else:
            if negative_polarity:
                literal_frequency[
                :self.clause_bank.number_of_literals // 2] += self.clause_bank.calculate_literal_clause_frequency(
                    self.weight_banks[the_class].get_weights() < 0)[:self.clause_bank.number_of_literals // 2]
            else:
                literal_frequency[
                :self.clause_bank.number_of_literals // 2] += self.clause_bank.calculate_literal_clause_frequency(
                    self.weight_banks[the_class].get_weights() >= 0)[:self.clause_bank.number_of_literals // 2]

        return literal_frequency

    def clause_precision_recall(self, the_class, positive_polarity, X, number_of_examples = 2000):
        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        X_csc = csc_matrix(X.reshape(X.shape[0], -1)).sorted_indices()

        if not np.array_equal(self.X_test, np.concatenate((X_csr.indptr, X_csr.indices))):
            self.encoded_X_test = self.clause_bank.prepare_X_autoencoder(X_csr, X_csc, self.output_active)
            self.X_test = np.concatenate((X_csr.indptr, X_csr.indices))

        #true_positive is array of size number of clauses and each element is the count of positive clauses 
        true_positive = np.zeros(self.number_of_clauses, dtype=np.uint32)
        false_negative = np.zeros(self.number_of_clauses, dtype=np.uint32)
        false_positive = np.zeros(self.number_of_clauses, dtype=np.uint32)

        weights = self.weight_banks[the_class].get_weights()

        for e in range(number_of_examples):
            Xu, Yu = self.clause_bank.produce_autoencoder_example_per_class(self.encoded_X_test, 
                                                                  the_class, 
                                                                  self.accumulation, 
                                                                  target_true_p= self.feature_true_probability[self.output_active[the_class]])
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(Xu, 0)

            if positive_polarity:
                if Yu == 1:
                    true_positive += (weights >= 0) * clause_outputs
                    false_negative += (weights >= 0) * (1 - clause_outputs)
                else:
                    false_positive += (weights >= 0) * clause_outputs
            else:
                if Yu == 0:
                    true_positive += (weights < 0) * clause_outputs
                    false_negative += (weights < 0) * (1 - clause_outputs)
                else:
                    false_positive += (weights < 0) * clause_outputs

        precision = 1.0 * true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        return precision, recall
    
    def get_weight(self, the_class, clause):
        return self.weight_banks[the_class].get_weights()[clause]

    def get_weights(self, the_class):
        return self.weight_banks[the_class].get_weights()

    def set_weight(self, the_class, clause, weight):
        self.weight_banks[the_class].get_weights()[clause] = weight
