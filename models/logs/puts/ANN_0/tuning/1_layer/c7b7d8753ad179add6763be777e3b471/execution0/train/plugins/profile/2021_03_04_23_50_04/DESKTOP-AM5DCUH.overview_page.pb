�	t���2@t���2@!t���2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-t���2@�	ܺ��0@1�cx�g�?A^K�=��?I��	��?*	fffff�I@2U
Iterator::Model::ParallelMapV2�Pk�w�?!���s;@)�Pk�w�?1���s;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapŏ1w-!�?!�Kh/�=@)46<�R�?1��O`?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!���
�+8@)��ZӼ�?1!����3@:Preprocessing2F
Iterator::ModelDio��ɔ?!��Fr�C@)-C��6z?18��<��(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice"��u��q?!:��0�� @)"��u��q?1:��0�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipX9��v��?!n#���6N@)_�Q�k?1,�����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!v�Il'@)/n��b?1v�Il'@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIq����X@Qd�~��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�	ܺ��0@�	ܺ��0@!�	ܺ��0@      ��!       "	�cx�g�?�cx�g�?!�cx�g�?*      ��!       2	^K�=��?^K�=��?!^K�=��?:	��	��?��	��?!��	��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qq����X@yd�~��?�"5
sequential/dense/MatMulMatMul�{�m+7�?!�{�m+7�?0"C
%gradient_tape/sequential/dense/MatMulMatMul���sĹ�?!J���w��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMuly�4��?!l�]I��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradå�y�Y�?!$/�瀈�?"7
sequential/dense_1/MatMulMatMul~i�)�?!T�F��?0"!
Adam/PowPowiQ�a��?!�F+)���?"E
'gradient_tape/sequential/dense_1/MatMulMatMuliQ�a��?!�p[5l��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�I8�~�?!�<����?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam� Irϒ?!�L3��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam� Irϒ?!�\�0��?Q      Y@Y�M�_{4@a��(�S@q�!V�XpX@yg�C�"d�?"�
both�Your program is POTENTIALLY input-bound because 91.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�7.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 