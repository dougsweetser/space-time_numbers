
# coding: utf-8

# # Lecture N:

# Load the needed libraries.

# In[5]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\nimport unittest\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# Make a class and a test library to work on.

# In[6]:


class QHStatesD(qt.QH):
    """A class made up of many quaternions."""
    
    def __init__(self, qs=None, qtype="", representation=""):
        pass


# In[8]:


class TestQHStatesD(unittest.TestCase):
    """Test states."""
    
    q_0 = qt.QH().q_0()
    
    def test_init(self):
        self.assertTrue(1)
    
suite = unittest.TestLoader().loadTestsFromModule(TestQHStatesD())
unittest.TextTestRunner().run(suite);

