	�sѐ�5@�sѐ�5@!�sѐ�5@	w�Kq!�?w�Kq!�?!w�Kq!�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�sѐ�5@%;6��2@1c��*3��?AA��ǘ��?I��o'�@Y�Z^��6s?*	������J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate8��d�`�?!�XQ�B@);�O��n�?1��/Ċ�@@:Preprocessing2U
Iterator::Model::ParallelMapV2vq�-�?!�rp�_�=@)vq�-�?1�rp�_�=@:Preprocessing2F
Iterator::ModelǺ���?!R�@�D@)F%u�{?1����B�(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��H�}}?!��L�w�*@)��_vOv?1�k9��/$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�U���؟?!���f�M@)Ǻ���f?1R�@�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��L�w�
@)��H�}]?1��L�w�
@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��_�LU?!�S�rp@)��_�LU?1�S�rp@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap^K�=��?!�ީk9�C@)a2U0*�S?1�_���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�_����?)a2U0*�C?1�_����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9w�Kq!�?I�3�|��X@Q��[�8��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	%;6��2@%;6��2@!%;6��2@      ��!       "	c��*3��?c��*3��?!c��*3��?*      ��!       2	A��ǘ��?A��ǘ��?!A��ǘ��?:	��o'�@��o'�@!��o'�@B      ��!       J	�Z^��6s?�Z^��6s?!�Z^��6s?R      ��!       Z	�Z^��6s?�Z^��6s?!�Z^��6s?b      ��!       JGPUYw�Kq!�?b q�3�|��X@y��[�8��?