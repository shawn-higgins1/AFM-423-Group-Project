	+����7@+����7@!+����7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-+����7@�ן��~4@1�xxρ��?A+��Χ?I����Q@*	33333�G@2U
Iterator::Model::ParallelMapV2S�!�uq�?!Da�-��;@)S�!�uq�?1Da�-��;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!����#�9@)��_�L�?1�^�?�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;�O��?!�4F#8@)���Q�~?1\���P/@:Preprocessing2F
Iterator::Model�j+��ݓ?!N3`4"@D@)�~j�t�x?1�
�u\)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!v\��� @)����Mbp?1v\��� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�v��/�?!�̟�ݿM@)-C��6j?1�`(�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!v\���@)����Mb`?1v\���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�{�Pk�?!~�e�\�:@)Ǻ���V?1����a@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��vK�X@Q|�E"���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ן��~4@�ן��~4@!�ן��~4@      ��!       "	�xxρ��?�xxρ��?!�xxρ��?*      ��!       2	+��Χ?+��Χ?!+��Χ?:	����Q@����Q@!����Q@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��vK�X@y|�E"���?