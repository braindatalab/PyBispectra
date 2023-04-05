.. _api_cfc:

Cross-Frequency Coupling
========================

:py:mod:`pybispectra.cfc`:

    .. automodule:: pybispectra.cfc
    
        {% block attributes %}
        {% if attributes %}
        .. rubric:: Module Attributes

        .. autosummary::
            :toctree:
        {% for item in attributes %}
            {{ item }}
        {%- endfor %}
        {% endif %}
        {% endblock %}

        {% block functions %}
        {% if functions %}
        .. rubric:: {{ _('Functions') }}

        .. autosummary::
            :toctree:
            :nosignatures:
        {% for item in functions %}
            {{ item }}
        {%- endfor %}
        {% endif %}
        {% endblock %}

        {% block classes %}
        {% if classes %}
        .. rubric:: {{ _('Classes') }}

        .. autosummary::
            :toctree:
            :template: custom-class-template.rst
        {% for item in classes %}
            {{ item }}
        {%- endfor %}
        {% endif %}
        {% endblock %}

        {% block exceptions %}
        {% if exceptions %}
        .. rubric:: {{ _('Exceptions') }}

        .. autosummary::
            :toctree:
        {% for item in exceptions %}
            {{ item }}
        {%- endfor %}
        {% endif %}
        {% endblock %}

    {% block modules %}
    {% if modules %}
    .. rubric:: Modules

.. autosummary::
    :toctree: generated/

    PAC
    PPC