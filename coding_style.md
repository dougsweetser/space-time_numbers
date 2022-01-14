# Coding Style

A core goal is to create simple, reliable, readible code. Classes will be the
home for space-time numbers and space-time number series. Functions can take
instances of these classes and do their thing.

To ease integration with software tools, the functions will be fully declared
and always have a __doc__ sting like so:

```
def add(q_1: Q, q_2: Q) -> Q:
    """
    Add two space-time numbers.

    $ q.add(q_2) = q_1 + q_2 = (t + t_2, R + R_2) $

    Args:
        q_1: Q
        q_2: Q

    Returns: Q

    """

```

More space is a good thing.

The difference between the function add() and adds() is the one with an
extra "s" acts of space-time number series.

## pep8 and pycodestyle

Apparently the python program pep8 is going to be replaced with
pycodestyle. Either way, all checked in code needs to pass both.
The code should look consistent no matter who the author happens
to be.

## Testing

pytest must pass 100% of the tests before any work is pushed to the main
branch. A goal of 100% test coverage will be worked on. Here is how these
programs work:

```
> coverage run Qs.py
> coverage report --show-missing
Name    Stmts   Miss  Cover   Missing
------------------------------------
Qs.py    1241   1069    14%   45, 50-58, 70, 89-109, ...
------------------------------------
TOTAL    1241   1069    14%
> pytest --ignore Archive --cov QS 
========================== test session starts ========================
platform darwin -- Python 3.9.7, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: /Volumes/ssd/Github/space-time_numbers, configfile: pytest.ini
plugins: anyio-2.2.0, cov-3.0.0
collected 133 items

test_Qs.py .....................................................[100%]

---------- coverage: platform darwin, python 3.9.7-final-0 -----------
Name    Stmts   Miss  Cover
---------------------------
Qs.py    1241    254    80%
---------------------------
TOTAL    1241    254    80%

========================== 133 passed in 3.58s =======================
>
```

