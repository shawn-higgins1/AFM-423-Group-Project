	3NCT��7@3NCT��7@!3NCT��7@	qF35wY�?qF35wY�?!qF35wY�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails63NCT��7@]�Fxo5@1�D��]�?A�H�}�?I
H�`�@Y�kC�8s?*	     �I@2U
Iterator::Model::ParallelMapV2�]K�=�?!:@)�]K�=�?1:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!;@)�+e�X�?1[ZZZZZ6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate9��v���?!}}}}}}9@)�&S��?1������1@:Preprocessing2F
Iterator::Model�ݓ��Z�?!������B@)Ǻ���v?1������%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8gDio�?!xxxxxxO@)�q����o?1������@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!������@)�q����o?1������@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!������@)a2U0*�c?1������@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��H�}�?!<<<<<<<@)Ǻ���V?1������@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9qF35wY�?I���g�X@Q������?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	]�Fxo5@]�Fxo5@!]�Fxo5@      ��!       "	�D��]�?�D��]�?!�D��]�?*      ��!       2	�H�}�?�H�}�?!�H�}�?:	
H�`�@
H�`�@!
H�`�@B      ��!       J	�kC�8s?�kC�8s?!�kC�8s?R      ��!       Z	�kC�8s?�kC�8s?!�kC�8s?b      ��!       JGPUYqF35wY�?b q���g�X@y������?