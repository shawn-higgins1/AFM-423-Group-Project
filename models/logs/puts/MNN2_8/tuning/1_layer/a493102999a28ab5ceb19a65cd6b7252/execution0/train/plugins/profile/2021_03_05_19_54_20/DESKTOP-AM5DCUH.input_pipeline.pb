	]�� P@]�� P@!]�� P@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-]�� P@�Wt�5�N@1��8����?A��d��J�?I�)x
9@*	33333�G@2U
Iterator::Model::ParallelMapV2��<,Ԋ?!��;@)��<,Ԋ?1��;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!��;@)��_vO�?1�=Q��6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�$��?!M����&6@)S�!�uq{?15+8|!E,@:Preprocessing2F
Iterator::Model�j+��ݓ?!C����vD@) �o_�y?1)�3�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!g�/� @)ŏ1w-!o?1g�/� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy�&1��?!�ur.�M@)��_vOf?1�=Q��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!|!E��h@)HP�s�b?1|!E��h@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!%����9@)_�Q�[?1��O�%�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 96.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIP����X@Q}�Y�E�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Wt�5�N@�Wt�5�N@!�Wt�5�N@      ��!       "	��8����?��8����?!��8����?*      ��!       2	��d��J�?��d��J�?!��d��J�?:	�)x
9@�)x
9@!�)x
9@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qP����X@y}�Y�E�?