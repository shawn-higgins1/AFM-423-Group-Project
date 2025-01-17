�	^&���5@^&���5@!^&���5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-^&���5@FB[Υ�3@1R���T�?A�e��a��?I���4� @*	43333sG@2U
Iterator::Model::ParallelMapV2_�Qڋ?!!7�Ct�<@)_�Qڋ?1!7�Ct�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!8��iF:@)M�O��?1Z�4��5@:Preprocessing2F
Iterator::Model�0�*��?!۶m۶mE@)9��v��z?1+mX��+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate��_�L�?!VW�Q�,6@) �o_�y?12����*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice	�^)�p?!z���{!@)	�^)�p?1z���{!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!$I�$I�L@)�~j�t�h?1<��J�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!v��@)�J�4a?1v��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!B�Ż8@)a2U0*�S?1c�l�x@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���X@Q����7��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	FB[Υ�3@FB[Υ�3@!FB[Υ�3@      ��!       "	R���T�?R���T�?!R���T�?*      ��!       2	�e��a��?�e��a��?!�e��a��?:	���4� @���4� @!���4� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���X@y����7��?�"5
sequential/dense/MatMulMatMult*) �'�?!t*) �'�?0"C
%gradient_tape/sequential/dense/MatMulMatMulgв�9��?!n�큍޵?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���\���?!���/M�?"7
sequential/dense_1/MatMulMatMul���\���?!����]�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamgв�9��?!�9����?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchgв�9��?!�Q7��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradgв�9��?!�큍��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�8���ߎ?!������?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�8���ߎ?!u3h���?"!
Adam/PowPow�8���ߎ?!�8�����?Q      Y@Y��/Ċ�0@a�	�N]�T@q���7|X@yв�9��?"�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 