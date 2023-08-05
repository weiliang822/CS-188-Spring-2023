import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        # 初始化权重
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # 计算感知器的输出
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        # 批大小为1
        batch_size = 1

        while 1:
            convergence = 1
            for x, y in dataset.iterate_once(batch_size):
                # 与预期不符
                if self.get_prediction(x) != nn.as_scalar(y):
                    convergence = 0
                    # 更新权值
                    self.w.update(x, nn.as_scalar(y))
            # 若收敛则退出循环
            if convergence:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 400    # 隐藏层大小
        self.batch_size = 100           # 批量大小，dataset的大小要能被batch_size整除
        self.learning_rate = 0.03       # 学习率
        self.input_size = 1             # 输入大小，sin函数只有一个参数
        self.output_size = 1            # 输出大小
        
        # 设一个隐藏层，共两个线性层
        '''
        x : batch_size x input_size
        W1 : input_size x hidden_layer_size
        b1 : 1 x hidden_layer_size
        W2 : hidden_layer_size x output_size
        b2 : 1 x output_size
        '''
        self.W1 = nn.Parameter(self.input_size,self.hidden_layer_size)
        # b的行为1，列与要加的矩阵相同
        self.b1 = nn.Parameter(1,self.hidden_layer_size)
        self.W2 = nn.Parameter(self.hidden_layer_size,self.output_size)
        self.b2 = nn.Parameter(1,self.output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # 隐藏层计算
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x,self.W1),self.b1))
        # 预期值
        prediction = nn.AddBias(nn.Linear(h1,self.W2),self.b2)
        return prediction

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while 1:
            for x,y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                # 计算损失梯度
                grad_W1,grad_b1,grad_W2,grad_b2 = nn.gradients(loss,[self.W1,self.b1,self.W2,self.b2])
                # 更新参数
                self.W1.update(grad_W1,-self.learning_rate)
                self.b1.update(grad_b1,-self.learning_rate)
                self.W2.update(grad_W2,-self.learning_rate)
                self.b2.update(grad_b2,-self.learning_rate)
            # 平均损失小于0.02时，训练完成
            if nn.as_scalar(loss) < 0.02:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 200    # 隐藏层大小
        self.batch_size = 100           # 批量大小，dataset的大小要能被batch_size整除
        self.learning_rate = 0.5        # 学习率
        self.input_size = 784           # 输入大小，28*28
        self.output_size = 10           # 输出大小

        # 设一个隐藏层，共两个线性层
        '''
        x : batch_size x input_size
        W1 : input_size x hidden_layer_size
        b1 : 1 x hidden_layer_size
        W2 : hidden_layer_size x output_size
        b2 : 1 x output_size
        '''
        self.W1 = nn.Parameter(self.input_size, self.hidden_layer_size)
        # b的行为1，列与要加的矩阵相同
        self.b1 = nn.Parameter(1, self.hidden_layer_size)
        self.W2 = nn.Parameter(self.hidden_layer_size, self.output_size)
        self.b2 = nn.Parameter(1, self.output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # 隐藏层计算
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        # 预期值
        prediction = nn.AddBias(nn.Linear(h1, self.W2), self.b2)
        return prediction

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while 1:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # 计算损失梯度
                grad_W1, grad_b1, grad_W2, grad_b2 = nn.gradients(
                    loss, [self.W1, self.b1, self.W2, self.b2])
                # 更新参数
                self.W1.update(grad_W1, -self.learning_rate)
                self.b1.update(grad_b1, -self.learning_rate)
                self.W2.update(grad_W2, -self.learning_rate)
                self.b2.update(grad_b2, -self.learning_rate)
            # 若准确率高于98%则退出
            if dataset.get_validation_accuracy() > 0.98:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 100                # 隐藏层大小
        self.batch_size = 100                       # 批量大小，dataset的大小要能被batch_size整除
        self.learning_rate = 0.45                   # 学习率
        self.input_size = self.num_chars            # 输入大小，47
        self.output_size = 5                        # 输出大小
      
        '''
        x : batch_size x input_size
        W_hidden : hidden_layer_size x hidden_layer_size
        Wx : input_size x hidden_layer_size
        Wo : hidden_layer_size x output_size
        '''
        self.W_hidden = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.Wx = nn.Parameter(self.input_size, self.hidden_layer_size)
        self.Wo = nn.Parameter(self.hidden_layer_size, self.output_size)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # print(len(xs))
        hi = nn.Linear(xs[0],self.Wx)
        h_hidden = nn.ReLU(hi)
        # 神经网络模拟
        for i in range(1,len(xs)):
            # xs[i]*Wx + h_hidden*W_hidden
            h_hidden = nn.ReLU(nn.Add(nn.Linear(xs[i],self.Wx),nn.Linear(h_hidden,self.W_hidden)))
        prediction = nn.Linear(h_hidden,self.Wo)
        return prediction

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while 1:
            for xs, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(xs, y)
                # 计算损失梯度
                grad_Wx, grad_W_hidden, grad_Wo = nn.gradients(
                    loss, [self.Wx, self.W_hidden, self.Wo])
                # 更新参数
                self.Wx.update(grad_Wx, -self.learning_rate)
                self.W_hidden.update(grad_W_hidden, -self.learning_rate)
                self.Wo.update(grad_Wo, -self.learning_rate)

            # 若准确率高于83%则退出
            if dataset.get_validation_accuracy() > 0.83:
                break