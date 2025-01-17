�	�M�»7@�M�»7@!�M�»7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�M�»7@_�Q.4@1�}�k�,�?AB`��"۩?IEdX��@*	33333�G@2U
Iterator::Model::ParallelMapV2��0�*�?!-���x�8@)��0�*�?1-���x�8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�&1��?!�a�3A:=@)��0�*�?1-���x�8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�$��?!Z����5@)9��v��z?1?����#+@:Preprocessing2F
Iterator::Model�:pΈ�?!eWC��B@) �o_�y?19u{N*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!v\��� @)����Mbp?1v\��� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2�%䃞?!�⨼AO@)_�Q�k?1ȶ��yd@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!����!_@)/n��b?1����!_@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��+e�?!����#�9@)ŏ1w-!_?1�b�?��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�U^�k�X@QX�j���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	_�Q.4@_�Q.4@!_�Q.4@      ��!       "	�}�k�,�?�}�k�,�?!�}�k�,�?*      ��!       2	B`��"۩?B`��"۩?!B`��"۩?:	EdX��@EdX��@!EdX��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�U^�k�X@yX�j���?�"5
sequential/dense/MatMulMatMul�\�_�?!�\�_�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�8��8�?!f�s)L�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�8��8�?!҈��x�?0"7
sequential/dense_1/MatMulMatMul�8��8�?!��d��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��(�?!��	���?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��pڇ�?!y�'���?"7
sequential/dense_2/MatMulMatMul.�Ɔ�?![t��e�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchc��p��?!gd����?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�8��8�?!�e����?")
sequential/CastCast�8��8�?!N��a%�?Q      Y@Y>����/@aX�i��U@q�PinX@y҈��x��?"�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 