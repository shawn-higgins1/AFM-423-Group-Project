	�Qԙ{2@�Qԙ{2@!�Qԙ{2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�Qԙ{2@F;�I�0@1࢓����?A�lV}��?Io.2^�?*	     J@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Q��?!��؉��<@)-C��6�?1ى�؉�8@:Preprocessing2U
Iterator::Model::ParallelMapV2�������?!�؉��	8@)�������?1�؉��	8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaplxz�,C�?!�؉�؉:@)n���?1��؉��2@:Preprocessing2F
Iterator::Model���&�?!�;��A@)a��+ey?1��؉��'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��镲�?!'vb'vP@);�O��nr?1��N��N!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!O��N��@)����Mbp?1O��N��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�N��N�@)/n��b?1�N��N�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIR�O��X@Q<W�y؀�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	F;�I�0@F;�I�0@!F;�I�0@      ��!       "	࢓����?࢓����?!࢓����?*      ��!       2	�lV}��?�lV}��?!�lV}��?:	o.2^�?o.2^�?!o.2^�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qR�O��X@y<W�y؀�?