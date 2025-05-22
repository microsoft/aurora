Application Programming Interface
=================================

Submission
----------
.. autofunction:: aurora.foundry.client.api.submit

.. autoclass:: aurora.foundry.client.foundry.FoundryClient
    :members: __init__

.. autoclass:: aurora.foundry.common.channel.BlobStorageChannel
    :members: __init__


Available Models
----------------
These models need to be referred to by the value of their attribute `name`.

.. autoclass:: aurora.foundry.common.model.AuroraFineTuned
    :members: name

.. autoclass:: aurora.foundry.common.model.AuroraPretrained
    :members: name

.. autoclass:: aurora.foundry.common.model.AuroraSmallPretrained
    :members: name

.. autoclass:: aurora.foundry.common.model.Aurora12hPretrained
    :members: name

.. autoclass:: aurora.foundry.common.model.AuroraHighRes
    :members: name

.. autoclass:: aurora.foundry.common.model.AuroraAirPollution
    :members: name

Server
------
.. autofunction:: aurora.foundry.server.mlflow_wrapper.AuroraModelWrapper
