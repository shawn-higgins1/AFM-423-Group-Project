�	�ތ��3@�ތ��3@!�ތ��3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�ތ��3@�4�O�0@1-$`ty�?A46<�R�?I�б�� @*	     �F@2U
Iterator::Model::ParallelMapV2F%u��?!.�-�=@)F%u��?1.�-�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!m۶m۶9@)�j+��݃?1�Q�Q5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ����?!ى�؉�8@)S�!�uq{?16Ws5Ws-@:Preprocessing2F
Iterator::Modela2U0*��?!�Q�QE@)�~j�t�x?1��_��_*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice;�O��nr?!|��{��#@);�O��nr?1|��{��#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	�c�?!j��j��L@)_�Q�k?1>��=��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!R�Q�@)����Mb`?1R�Q�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI.�$뱩X@Qgt�6���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�4�O�0@�4�O�0@!�4�O�0@      ��!       "	-$`ty�?-$`ty�?!-$`ty�?*      ��!       2	46<�R�?46<�R�?!46<�R�?:	�б�� @�б�� @!�б�� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q.�$뱩X@ygt�6���?�"5
sequential/dense/MatMulMatMul�z�͍�?!�z�͍�?0"C
%gradient_tape/sequential/dense/MatMulMatMul���C�a�?!+����?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�k�� ��?!�.�s�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�k�� ��?!\)Zj�?"7
sequential/dense_1/MatMulMatMul�k�� ��?!���C�a�?0")
sequential/CastCast���C�a�?!�z�͍�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul��a��
�?!����&��?0"D
&mean_squared_error/weighted_loss/valueDivNoNan��a��
�?!�y��?��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam.���3	�?!
T�:���?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam.���3	�?!}.�uf	�?Q      Y@Y�M�_{4@a��(�S@q&����wX@yz�ڄ@��?"�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 