# Combining all the pieces together
class PytorchBasedGenericGradientBoost():
    def __init__(self, type, n_trees, max_depth, GRADIENT_BOOST_LEARNING_RATE = 0.1, MINIMIZER_LEARNING_RATE = 0.001, MINIMIZER_TRAINING_EPOCHS = 5000):
        '''
        type : "regressor" or "classifier"
        '''
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.type = type
        self.gradient_boost_learning_rate = GRADIENT_BOOST_LEARNING_RATE
        self.minimizer_learning_rate = MINIMIZER_LEARNING_RATE
        self.minimizer_training_epochs = MINIMIZER_TRAINING_EPOCHS
        # Variables to hold output of algorithm
        self.initial_prediction = None
        self.regression_trees = []
        # Get an instance of a minimizer
        self.minimizer = LossFunctionMinimizer(self.type)
        if USE_CUDA:
            self.minimizer.cuda()
        self.minimizer_optimizer = torch.optim.Adam(self.minimizer.parameters(), lr=self.minimizer_learning_rate)
    def minimize_loss_function(self, targets, previous_predictions):
        self.minimizer.reinitialize_variable()
        for training_epoch in range(self.minimizer_training_epochs):
            targets_leaf_tensor = FloatTensor(targets)
            loss = self.minimizer.loss_classifier(previous_predictions, targets_leaf_tensor)
            self.minimizer.zero_grad()
            loss.backward()
            self.minimizer_optimizer.step()
        return [el for el in self.minimizer.parameters()][0].cpu().detach().numpy()[0]
    def compute_residuals(self, targets, predicted_values):
        model = ResidualsCalculator(predicted_values, self.type)
        if USE_CUDA:
            model.cuda()
        loss = model.loss(targets)
        model.zero_grad()
        loss.backward()
        residuals = model.predicted_values.grad.clone() # deep copy of the input/gradients
        return residuals
    def fit(self, X, y):
        X_values = X.copy()
        y_values = y.copy()
        # Initialization phase
        if USE_CUDA:
            initial_values = torch.zeros(y_values.shape,1).cuda()
        else:
            initial_values = torch.zeros(y_values.shape)
        self.initial_prediction = self.minimize_loss_function(y_values, initial_values)
        prediction_values = np.ones(y_values.shape) * self.initial_prediction

        for classifier_index in range(self.n_trees):
            self.regression_trees.append({"tree_index": classifier_index})
            residuals = self.compute_residuals(FloatTensor(y_values), FloatTensor(prediction_values))
            leaf_buckets, unique_clusters, tree_regressor = fit_regression_tree_classifier_to_residuals(X_values, residuals.cpu(), self.max_depth)
            self.regression_trees[-1]["tree_regressor"] = tree_regressor
            
            X_values_temp = np.array([])
            y_values_temp = np.array([])
            prediction_values_temp = np.array([])

            for unique_cluster in unique_clusters:
                indices = [1 if el == unique_cluster else 0 for el in leaf_buckets]
                y_leaf = y_values[np.array(indices) == 1]
                X_leaf = X_values[np.array(indices) == 1]
                predictions_leaf = prediction_values[np.array(indices) == 1]
                prediction_for_leaf = self.minimize_loss_function(FloatTensor(np.array(y_leaf)), FloatTensor(predictions_leaf))
                predictions_for_leaf_array = np.ones(y_leaf.shape) * self.gradient_boost_learning_rate * prediction_for_leaf + predictions_leaf
                self.regression_trees[-1][str(unique_cluster)] = prediction_for_leaf
                X_values_temp = X_leaf if X_values_temp.shape == (0, ) else np.append(X_values_temp, X_leaf, axis=0)
                y_values_temp = np.append(y_values_temp, y_leaf)
                prediction_values_temp = np.append(prediction_values_temp, predictions_for_leaf_array)
            y_values = y_values_temp
            X_values = X_values_temp
            prediction_values = prediction_values_temp    
    def predict(self, X):
        predictions = []
        for index in range(X.shape[0]):
            prediction = self.initial_prediction
            for tree_index in range(self.n_trees):
                tree = self.regression_trees[tree_index]
                prediction += self.gradient_boost_learning_rate * tree[str(tuple(tree["tree_regressor"].decision_path(X[index, :].reshape(1,-1)).todok().keys()))]
            predictions.append(prediction)
        if self.type == "regressor":
            return predictions
        elif self.type == "classifier":
            return torch.sigmoid(torch.tensor(predictions)).numpy()
        else:
            raise Exception("Not supported")# Combining all the pieces together
class PytorchBasedGenericGradientBoost():
    def __init__(self, type, n_trees, max_depth, GRADIENT_BOOST_LEARNING_RATE = 0.1, MINIMIZER_LEARNING_RATE = 0.001, MINIMIZER_TRAINING_EPOCHS = 5000):
        '''
        type : "regressor" or "classifier"
        '''
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.type = type
        self.gradient_boost_learning_rate = GRADIENT_BOOST_LEARNING_RATE
        self.minimizer_learning_rate = MINIMIZER_LEARNING_RATE
        self.minimizer_training_epochs = MINIMIZER_TRAINING_EPOCHS
        # Variables to hold output of algorithm
        self.initial_prediction = None
        self.regression_trees = []
        # Get an instance of a minimizer
        self.minimizer = LossFunctionMinimizer(self.type)
        if USE_CUDA:
            self.minimizer.cuda()
        self.minimizer_optimizer = torch.optim.Adam(self.minimizer.parameters(), lr=self.minimizer_learning_rate)
    def minimize_loss_function(self, targets, previous_predictions):
        self.minimizer.reinitialize_variable()
        for training_epoch in range(self.minimizer_training_epochs):
            targets_leaf_tensor = FloatTensor(targets)
            loss = self.minimizer.loss_classifier(previous_predictions, targets_leaf_tensor)
            self.minimizer.zero_grad()
            loss.backward()
            self.minimizer_optimizer.step()
        return [el for el in self.minimizer.parameters()][0].cpu().detach().numpy()[0]
    def compute_residuals(self, targets, predicted_values):
        model = ResidualsCalculator(predicted_values, self.type)
        if USE_CUDA:
            model.cuda()
        loss = model.loss(targets)
        model.zero_grad()
        loss.backward()
        residuals = model.predicted_values.grad.clone() # deep copy of the input/gradients
        return residuals
    def fit(self, X, y):
        X_values = X.copy()
        y_values = y.copy()
        # Initialization phase
        if USE_CUDA:
            initial_values = torch.zeros(y_values.shape,1).cuda()
        else:
            initial_values = torch.zeros(y_values.shape)
        self.initial_prediction = self.minimize_loss_function(y_values, initial_values)
        prediction_values = np.ones(y_values.shape) * self.initial_prediction

        for classifier_index in range(self.n_trees):
            self.regression_trees.append({"tree_index": classifier_index})
            residuals = self.compute_residuals(FloatTensor(y_values), FloatTensor(prediction_values))
            leaf_buckets, unique_clusters, tree_regressor = fit_regression_tree_classifier_to_residuals(X_values, residuals.cpu(), self.max_depth)
            self.regression_trees[-1]["tree_regressor"] = tree_regressor
            
            X_values_temp = np.array([])
            y_values_temp = np.array([])
            prediction_values_temp = np.array([])

            for unique_cluster in unique_clusters:
                indices = [1 if el == unique_cluster else 0 for el in leaf_buckets]
                y_leaf = y_values[np.array(indices) == 1]
                X_leaf = X_values[np.array(indices) == 1]
                predictions_leaf = prediction_values[np.array(indices) == 1]
                prediction_for_leaf = self.minimize_loss_function(FloatTensor(np.array(y_leaf)), FloatTensor(predictions_leaf))
                predictions_for_leaf_array = np.ones(y_leaf.shape) * self.gradient_boost_learning_rate * prediction_for_leaf + predictions_leaf
                self.regression_trees[-1][str(unique_cluster)] = prediction_for_leaf
                X_values_temp = X_leaf if X_values_temp.shape == (0, ) else np.append(X_values_temp, X_leaf, axis=0)
                y_values_temp = np.append(y_values_temp, y_leaf)
                prediction_values_temp = np.append(prediction_values_temp, predictions_for_leaf_array)
            y_values = y_values_temp
            X_values = X_values_temp
            prediction_values = prediction_values_temp    
    def predict(self, X):
        predictions = []
        for index in range(X.shape[0]):
            prediction = self.initial_prediction
            for tree_index in range(self.n_trees):
                tree = self.regression_trees[tree_index]
                prediction += self.gradient_boost_learning_rate * tree[str(tuple(tree["tree_regressor"].decision_path(X[index, :].reshape(1,-1)).todok().keys()))]
            predictions.append(prediction)
        if self.type == "regressor":
            return predictions
        elif self.type == "classifier":
            return torch.sigmoid(torch.tensor(predictions)).numpy()
        else:
            raise Exception("Not supported")