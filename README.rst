AirProfile
==========

.. image:: https://badge.fury.io/py/AirProfile.svg
    :target: https://badge.fury.io/py/AirProfile

A python package for automatic analysis of Airbnb host profiles.

The package takes an Airbnb profile, automatically tags topics for each sentence, and predicts whether the profile will be perceived as more trustworthy compared to other profiles of similar length.

Installation
------------

.. code:: bash

  pip install AirProfile

A mirrored package is available on Conda via ``kenlimmj/AirProfile``. However, updates are on a best effort basis, and you should use PyPI where possible.

Example Usage
-------------
`LIWC2007 <https://liwc.wpengine.com>`_ is a paid third-party dependency. You'll need your own copy in order to predict trust, but you can still predict topics without it.

.. code:: python

  >>> from AirProfile import AirProfile
  >>> ap = AirProfile(liwc_path='../LIWC2007/liwc_2007.trie')
  # or `ap = AirProfile()`, if you do not have LIWC.

  # Example Airbnb host profile.
  >>> input = "I have spent my life in the service industry." \
      "I look forward to being your host and I look forward to meeting you."

  # Segments the input at the sentence level and returns the probability that
  # each sentence is tagged with the topics described in [1]. This works with or
  # without LIWC.
  >>> ap.predict_topics(input)
  [
    [
      'i have spent my life in the service industry',
      {
        'relationships': 0.02,
        'workEducation': 0.99,
        'travel': 0.0,
        'originResidence': 0.07,
        'lifeMottoValues': 0.03,
        'hospitality': 0.02,
        'interestsTastes': 0.03,
        'personality': 0.02
      }
    ], [
      'i look forward to being your host and i look forward to meeting you',
      {
        'relationships': 0.0,
        'workEducation': 0.0,
        'travel': 0.02,
        'originResidence': 0.0,
        'lifeMottoValues': 0.0,
        'hospitality': 1.0,
        'interestsTastes': 0.0,
        'personality': 0.04
      }
    ]
  ]

  # Segments the input at the sentence level and returns the probability that
  # the profile is perceived to be more trustworthy compared to other profiles
  # of similar length. This requires LIWC and will throw an error otherwise.
  >>> ap.predict_trust(input)
  Prediction(prob=0.49, predict=0)

Versioning
----------
Development will be maintained under the Semantic Versioning guidelines as much as possible in order to ensure transparency and backward compatibility.

Releases will be numbered with the following format::

<major>.<minor>.<patch>

And constructed with the following guidelines:

- Breaking backward compatibility bumps the major (and resets the minor and
  patch).
- New additions without breaking backward compatibility bump the minor (and
  resets the patch).
- Bug fixes and miscellaneous changes bump the patch.

For more information on SemVer, visit http://semver.org.

Bug Tracking and Feature Requests
---------------------------------
Have a bug or a feature request? `Please open a new issue <https://github.com/sTechLab/AirProfile/issues>`_.

Before opening any issue, please search for existing issues and read the `Issue Guidelines <https://github.com/sTechLab/AirProfile/blob/master/CONTRIBUTING.md>`_.

Contributing
------------
Please submit all pull-requests against ``*-wip`` branches. Code should adhere to the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_ and be linted using `Yapf <https://github.com/google/yapf>`_.

References
----------
[1] Self-disclosure and Perceived Trustworthiness of Airbnb Host Profiles. Xiao Ma, Jeff Hancock, Kenneth Lim Mingjie, and Mor Naaman. CSCW 2017. Honorable Mention for Best Paper. [PDF1_]

.. _PDF1: https://s.tech.cornell.edu/assets/papers/ma2017airbnb.pdf

[2] A Computational Approach to Perceived Trustworthiness of Airbnb Host Profiles. Xiao Ma, Trishala Neeraj, Mor Naamann. ICWSM 2017. Poster. [PDF2_]

.. _PDF2: http://maxiao.info/assets/computational-airbnb.pdf
