	��yr}3@��yr}3@!��yr}3@	��Hr#�?��Hr#�?!��Hr#�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��yr}3@�h���1@1�#)�ah�?A�E���Ԩ?I1DN_�W�?Y9~�4bfo?*	43333sK@2U
Iterator::Model::ParallelMapV2�5�;Nё?!�D��?@)�5�;Nё?1�D��?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�q����?!f�"Qj<@)��@��ǈ?1�7B
6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�^ѓ��6@)��ZӼ�?1�Y@�H�2@:Preprocessing2F
Iterator::Model=�U����?!�l���E@)S�!�uq{?1�Ϛ�sh(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!.�˯;�@)y�&1�l?1.�˯;�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip� �	��?! Z�tL@)��_vOf?1�Oq��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�DV��@)HP�s�b?1�DV��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��Hr#�?I� Y��X@QtP���K�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�h���1@�h���1@!�h���1@      ��!       "	�#)�ah�?�#)�ah�?!�#)�ah�?*      ��!       2	�E���Ԩ?�E���Ԩ?!�E���Ԩ?:	1DN_�W�?1DN_�W�?!1DN_�W�?B      ��!       J	9~�4bfo?9~�4bfo?!9~�4bfo?R      ��!       Z	9~�4bfo?9~�4bfo?!9~�4bfo?b      ��!       JGPUY��Hr#�?b q� Y��X@ytP���K�?