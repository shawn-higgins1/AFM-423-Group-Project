	����C5@����C5@!����C5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����C5@��9��2@1�����?A��_vO�?I�U��y� @*	23333sG@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��d�`T�?!�N�IC@)L7�A`�?1�Xi�2�A@:Preprocessing2U
Iterator::Model::ParallelMapV2g��j+��?!@b�Z�8@)g��j+��?1@b�Z�8@:Preprocessing2F
Iterator::Model46<��?!�k�t�C@)F%u�{?1([u%,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t�x?!<��J�)@)�J�4q?1v��!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���QI�?!x�0�}N@)F%u�k?1([u%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�"� �@)��H�}]?1�"� �@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�N@aÓ?!��3�`�D@)Ǻ���V?1HT�n�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!.;��J�?)-C��6J?1.;��J�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!c�l�x�?)a2U0*�C?1c�l�x�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI3=6�X@Q�@f��d�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��9��2@��9��2@!��9��2@      ��!       "	�����?�����?!�����?*      ��!       2	��_vO�?��_vO�?!��_vO�?:	�U��y� @�U��y� @!�U��y� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q3=6�X@y�@f��d�?