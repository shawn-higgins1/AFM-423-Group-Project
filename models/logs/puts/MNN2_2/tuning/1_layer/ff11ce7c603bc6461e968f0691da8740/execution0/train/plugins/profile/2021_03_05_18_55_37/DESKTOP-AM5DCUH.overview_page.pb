�	�D�$IP@�D�$IP@!�D�$IP@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�D�$IP@̛õ�CO@1��w�-;�?A$����ۧ?I����@*	gffff�F@2U
Iterator::Model::ParallelMapV2�HP��?!s���6�:@)�HP��?1s���6�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!5��̕�;@)�0�*�?1���4}6@:Preprocessing2F
Iterator::Model�ݓ��Z�?!x���D@)S�!�uq{?1�z*��A-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate�j+��݃?!H���-5@)a��+ey?1��.+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicey�&1�l?!�e��S�@)y�&1�l?1�e��S�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��q���?!��g�]M@)-C��6j?15��̕�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!�H���@){�G�zd?1�H���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!Ψ5���7@)a2U0*�S?1(�nY��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 96.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIk�I���X@QY�M�1"�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	̛õ�CO@̛õ�CO@!̛õ�CO@      ��!       "	��w�-;�?��w�-;�?!��w�-;�?*      ��!       2	$����ۧ?$����ۧ?!$����ۧ?:	����@����@!����@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qk�I���X@yY�M�1"�?�"5
sequential/dense/MatMulMatMulų��?!ų��?0"C
%gradient_tape/sequential/dense/MatMulMatMul޸�N�w�?!�>��Q��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulM��J��?!Q*_$l�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�M"i��?!��S���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch2T�C�?!�)��0�?"7
sequential/dense_1/MatMulMatMul޸�N�w�?!��U�!��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam���"���?!>\���?"!
Adam/PowPow����ٵ�?!�j��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamM��J��?!1�|0K��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamM��J��?!�M����?Q      Y@Y��/Ċ�0@a�	�N]�T@q>����X@y_���1<�?"�
both�Your program is POTENTIALLY input-bound because 96.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�96.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 