�	O���7@O���7@!O���7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-O���7@UMuC4@1��A^&�?A�,C��?IG� \e@*fffff&K@)       =2U
Iterator::Model::ParallelMapV2�q����?!q�5��<@)�q����?1q�5��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!BZ��7@)46<�R�?14p���4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate9��v���?!	st���7@)��~j�t�?1��D�~1@:Preprocessing2F
Iterator::Model��ͪ�Ֆ?!�pϸ�D@)S�!�uq{?1�D�~�(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����Mb�?!�R�0GwM@)��H�}m?1A�ME�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!1h}J^�@)y�&1�l?11h}J^�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�q����?!q�5��<@)��_�Le?1��)y!'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!a��:�@)ŏ1w-!_?1a��:�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�BC�X@QZC�>/x�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	UMuC4@UMuC4@!UMuC4@      ��!       "	��A^&�?��A^&�?!��A^&�?*      ��!       2	�,C��?�,C��?!�,C��?:	G� \e@G� \e@!G� \e@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�BC�X@yZC�>/x�?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul������?!������?"5
sequential/dense/MatMulMatMul��/�Z��?!��\��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�cm��?!��p��$�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�cm��?!{�	�dL�?0"7
sequential/dense_1/MatMulMatMul�cm��?!�q�0@t�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul`ֹ�c�?!H4ݧ� �?"7
sequential/dense_2/MatMulMatMul`ֹ�c�?!��#��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�cm��?!�3�_�P�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�cm��?!
lX0Z�?"&
	truediv_1RealDivu6w7�?!��"�?Q      Y@Y>����/@aX�i��U@q�k��mX@y����w�?"�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 