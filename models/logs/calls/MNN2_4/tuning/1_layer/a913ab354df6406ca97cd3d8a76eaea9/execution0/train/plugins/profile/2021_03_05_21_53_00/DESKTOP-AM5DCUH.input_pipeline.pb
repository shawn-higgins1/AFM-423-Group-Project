	�?��wh6@�?��wh6@!�?��wh6@	�B���l�?�B���l�?!�B���l�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�?��wh6@}�K���3@1�A��ފ�?A#�ng_y�?I�J %v@Y�����_z?*	53333sI@2U
Iterator::Model::ParallelMapV22�%䃎?!��K��E=@)2�%䃎?1��K��E=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq�?!���;}S:@)M�St$�?1�#m�L36@:Preprocessing2F
Iterator::Model�e��a��?!+ǄX��E@)��H�}}?1�8{�oJ,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�+e�X�?!T�d�e6@)��H�}}?1�8{�oJ,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!ҋ�!�� @)�J�4q?1ҋ�!�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��H�}�?!�8{�oJL@)�����g?1�_�F/�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!ҋ�!��@)�J�4a?1ҋ�!��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!bA֎8@)/n��R?1I�Y��I@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�B���l�?I�S0���X@Q�B��Ӝ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	}�K���3@}�K���3@!}�K���3@      ��!       "	�A��ފ�?�A��ފ�?!�A��ފ�?*      ��!       2	#�ng_y�?#�ng_y�?!#�ng_y�?:	�J %v@�J %v@!�J %v@B      ��!       J	�����_z?�����_z?!�����_z?R      ��!       Z	�����_z?�����_z?!�����_z?b      ��!       JGPUY�B���l�?b q�S0���X@y�B��Ӝ�?