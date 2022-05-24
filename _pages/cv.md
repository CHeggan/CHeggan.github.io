---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* MPhys, University Of Edinburgh, 2020
* Ph.D in Frugal Machine Learning, University Of Edinburgh, 2024 (expected)

Research Experience
======
* 2019: Senior Honours Project
  * Title: X-Ray Diffraction from Crystals with Realistic Charge Distributions
  * Duties included: Simulation of high pressure phase transitions of Magnesium using the Wien2k package
  * Supervisor: Ingo Loa

* 2020: Masters Project
  * Title: Electride Phases and Host-Guest lattices of Group V Elements
  * Duties included: Simulation work with the VASP package using a variety of high performance computing resources (ARCHER, Thomas)
  * Supervisor: Andreas Hermann
  
Skills
======
* Numerical & Machine Learning Frameworks
  * PyTorch
  * TensorFlow
  * Sklearn
  * SciPy
* Programming
  * Python
  * R
  * MATLAB
  * SQL

Publications
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>