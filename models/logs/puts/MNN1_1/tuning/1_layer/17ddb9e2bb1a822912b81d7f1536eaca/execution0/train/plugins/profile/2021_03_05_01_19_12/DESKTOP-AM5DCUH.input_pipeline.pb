	��!��5@��!��5@!��!��5@	-e�s?-e�s?!-e�s?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��!��5@�f��6/3@1���V�?A���_vO�?I��c5@Y������P?*	433333L@2U
Iterator::Model::ParallelMapV2� �	��?!�y�'N;@)� �	��?1�y�'N;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!�n��8@)�+e�X�?1��l�w64@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�ZӼ��?!��c-9@)��~j�t�?1��Z��0@:Preprocessing2F
Iterator::ModeltF��_�?!��kE@)�J�4�?1W�+��-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU���N@s?!������ @)U���N@s?1������ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���{�?!"��L@)-C��6j?1[����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!��	�4@)/n��b?1��	�4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapvq�-�?!�����<@)-C��6Z?1[����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9-e�s?I�8N$�X@Qn٭�c�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�f��6/3@�f��6/3@!�f��6/3@      ��!       "	���V�?���V�?!���V�?*      ��!       2	���_vO�?���_vO�?!���_vO�?:	��c5@��c5@!��c5@B      ��!       J	������P?������P?!������P?R      ��!       Z	������P?������P?!������P?b      ��!       JGPUY-e�s?b q�8N$�X@yn٭�c�?