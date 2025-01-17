�	E�^Ӄ:7@E�^Ӄ:7@!E�^Ӄ:7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-E�^Ӄ:7@��\4d�4@1�4D��?A(��y�?I^��vq @*	gffff�M@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�N@aÓ?!�̳�#@@)K�=�U�?1&|�0Օ9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9��v��?!�0Օs�9@)F%u��?1C���6@:Preprocessing2U
Iterator::Model::ParallelMapV29��v���?!����7�5@)9��v���?1����7�5@:Preprocessing2F
Iterator::Modela2U0*��?!w�q�@@)a��+ey?1���w\�$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipDio��ɤ?!DG�&�P@)	�^)�p?1��w\�l@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!qth��@)����Mbp?1qth��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!O��N��@)HP�s�b?1O��N��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/�$��?!j����A@)_�Q�[?1����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���ÜX@Q,y���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��\4d�4@��\4d�4@!��\4d�4@      ��!       "	�4D��?�4D��?!�4D��?*      ��!       2	(��y�?(��y�?!(��y�?:	^��vq @^��vq @!^��vq @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���ÜX@y,y���?�"5
sequential/dense/MatMulMatMul���>b�?!���>b�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�]A~l;�?!(�)��N�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�]A~l;�?!�1J���?0"7
sequential/dense_1/MatMulMatMul�]A~l;�?!^p�!��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul4$,��:�?!ky����?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch@-�P���?!_�1��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad@-�P���?!�D��@v�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul@-�P���?!c*�t'�?"7
sequential/dense_2/MatMulMatMul@-�P���?!��Sl�?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam4$,��:�?!I�p���?Q      Y@Y>����/@aX�i��U@q����]X@y��4%��?"�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 